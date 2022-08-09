import pyhf
from pyhf.optimize.mixins import OptimizerMixin
from pyhf.optimize.common import _get_tensor_shim, _make_stitch_pars
from pyhf.tensor.common import _TensorViewer
import pytest
from scipy.optimize import minimize, OptimizeResult
from scipy.optimize import OptimizeWarning
import iminuit
import itertools
import numpy as np


# from https://docs.scipy.org/doc/scipy/tutorial/optimize.html#nelder-mead-simplex-algorithm-method-nelder-mead
@pytest.mark.skip_pytorch
@pytest.mark.skip_pytorch64
@pytest.mark.skip_tensorflow
@pytest.mark.skip_numpy_minuit
def test_scipy_minimize(backend, capsys):
    tensorlib, _ = backend

    def rosen(x):
        """The Rosenbrock function"""
        return sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)

    x0 = tensorlib.astensor([1.3, 0.7, 0.8, 1.9, 1.2])
    res = minimize(rosen, x0, method='SLSQP', options=dict(disp=True))
    captured = capsys.readouterr()
    assert "Optimization terminated successfully" in captured.out
    assert pytest.approx([1.0, 1.0, 1.0, 1.0, 1.0], rel=5e-5) == tensorlib.tolist(res.x)


@pytest.mark.parametrize('do_stitch', [False, True], ids=['no_stitch', 'do_stitch'])
@pytest.mark.parametrize(
    'tensorlib',
    [
        pyhf.tensor.numpy_backend,
        pyhf.tensor.pytorch_backend,
        pyhf.tensor.tensorflow_backend,
        pyhf.tensor.jax_backend,
    ],
    ids=['numpy', 'pytorch', 'tensorflow', 'jax'],
)
@pytest.mark.parametrize(
    'optimizer',
    [pyhf.optimize.scipy_optimizer, pyhf.optimize.minuit_optimizer],
    ids=['scipy', 'minuit'],
)
@pytest.mark.parametrize('do_grad', [False, True], ids=['no_grad', 'do_grad'])
def test_minimize(tensorlib, optimizer, do_grad, do_stitch):
    pyhf.set_backend(tensorlib(precision="64b"), optimizer())
    m = pyhf.simplemodels.uncorrelated_background([50.0], [100.0], [10.0])
    data = pyhf.tensorlib.astensor([125.0] + m.config.auxdata)
    # numpy does not support grad
    if pyhf.tensorlib.name == 'numpy' and do_grad:
        with pytest.raises(pyhf.exceptions.Unsupported):
            pyhf.infer.mle.fit(data, m, do_grad=do_grad)
    else:
        identifier = f'{"do_grad" if do_grad else "no_grad"}-{pyhf.optimizer.name}-{pyhf.tensorlib.name}'
        expected = {
            # numpy does not do grad
            'do_grad-scipy-numpy': None,
            'do_grad-minuit-numpy': None,
            # no grad, scipy, 64b
            'no_grad-scipy-numpy': [0.49998815367220306, 0.9999696999038924],
            'no_grad-scipy-pytorch': [0.49998815367220306, 0.9999696999038924],
            'no_grad-scipy-tensorflow': [0.49998865164653106, 0.9999696533705097],
            'no_grad-scipy-jax': [0.4999880886490433, 0.9999696971774877],
            # do grad, scipy, 64b
            'do_grad-scipy-pytorch': [0.49998837853531425, 0.9999696648069287],
            'do_grad-scipy-tensorflow': [0.4999883785353142, 0.9999696648069278],
            'do_grad-scipy-jax': [0.49998837853531414, 0.9999696648069285],
            # no grad, minuit, 64b - quite consistent
            'no_grad-minuit-numpy': [0.5000493563629738, 1.0000043833598724],
            'no_grad-minuit-pytorch': [0.5000493563758468, 1.0000043833508256],
            'no_grad-minuit-tensorflow': [0.5000493563645547, 1.0000043833598657],
            'no_grad-minuit-jax': [0.5000493563528641, 1.0000043833614634],
            # do grad, minuit, 64b
            'do_grad-minuit-pytorch': [0.500049321728735, 1.00000441739846],
            'do_grad-minuit-tensorflow': [0.5000492930412292, 1.0000044107437134],
            'do_grad-minuit-jax': [0.500049321731032, 1.0000044174002167],
        }[identifier]

        result = pyhf.infer.mle.fit(data, m, do_grad=do_grad, do_stitch=do_stitch)

        rel_tol = 1e-5 if "no_grad" in identifier else 1e-6
        # Fluctuations beyond precision shouldn't matter
        abs_tol = 1e-8

        # check fitted parameters
        assert pytest.approx(
            expected, rel=rel_tol, abs=abs_tol
        ) == pyhf.tensorlib.tolist(
            result
        ), f"{identifier} = {pyhf.tensorlib.tolist(result)}"


@pytest.mark.parametrize(
    'optimizer',
    [OptimizerMixin, pyhf.optimize.scipy_optimizer, pyhf.optimize.minuit_optimizer],
    ids=['mixin', 'scipy', 'minuit'],
)
def test_optimizer_mixin_extra_kwargs(optimizer):
    with pytest.raises(pyhf.exceptions.Unsupported):
        optimizer(fake_kwarg=False)


@pytest.mark.parametrize(
    'backend,backend_new',
    itertools.permutations(
        [('numpy', False), ('pytorch', True), ('tensorflow', True), ('jax', True)], 2
    ),
    ids=lambda pair: f'{pair[0]}',
)
def test_minimize_do_grad_autoconfig(mocker, backend, backend_new):
    backend, do_grad = backend
    backend_new, do_grad_new = backend_new

    # patch all we need
    from pyhf.optimize import mixins

    shim = mocker.patch.object(mixins, 'shim', return_value=({}, lambda x: True))
    mocker.patch.object(OptimizerMixin, '_internal_minimize')
    mocker.patch.object(OptimizerMixin, '_internal_postprocess')

    # start with first backend
    pyhf.set_backend(backend, 'scipy')
    m = pyhf.simplemodels.uncorrelated_background([50.0], [100.0], [10.0])
    data = pyhf.tensorlib.astensor([125.0] + m.config.auxdata)

    assert pyhf.tensorlib.default_do_grad == do_grad
    pyhf.infer.mle.fit(data, m)
    assert shim.call_args[1]['do_grad'] == pyhf.tensorlib.default_do_grad
    pyhf.infer.mle.fit(data, m, do_grad=not (pyhf.tensorlib.default_do_grad))
    assert shim.call_args[1]['do_grad'] != pyhf.tensorlib.default_do_grad

    # now switch to new backend and see what happens
    pyhf.set_backend(backend_new)
    m = pyhf.simplemodels.uncorrelated_background([50.0], [100.0], [10.0])
    data = pyhf.tensorlib.astensor([125.0] + m.config.auxdata)

    assert pyhf.tensorlib.default_do_grad == do_grad_new
    pyhf.infer.mle.fit(data, m)
    assert shim.call_args[1]['do_grad'] == pyhf.tensorlib.default_do_grad
    pyhf.infer.mle.fit(data, m, do_grad=not (pyhf.tensorlib.default_do_grad))
    assert shim.call_args[1]['do_grad'] != pyhf.tensorlib.default_do_grad


def test_minuit_strategy_do_grad(mocker, backend):
    """
    ref: gh#1172

    When there is a user-provided gradient, check that one automatically sets
    the minuit strategy=0. When there is no user-provided gradient, check that
    one automatically sets the minuit strategy=1.
    """
    pyhf.set_backend(pyhf.tensorlib, pyhf.optimize.minuit_optimizer(tolerance=0.2))
    spy = mocker.spy(pyhf.optimize.minuit_optimizer, '_minimize')
    m = pyhf.simplemodels.uncorrelated_background([50.0], [100.0], [10.0])
    data = pyhf.tensorlib.astensor([125.0] + m.config.auxdata)

    do_grad = pyhf.tensorlib.default_do_grad
    pyhf.infer.mle.fit(data, m)
    assert spy.call_count == 1
    assert not spy.spy_return.minuit.strategy == do_grad

    pyhf.infer.mle.fit(data, m, strategy=0)
    assert spy.call_count == 2
    assert spy.spy_return.minuit.strategy == 0

    pyhf.infer.mle.fit(data, m, strategy=1)
    assert spy.call_count == 3
    assert spy.spy_return.minuit.strategy == 1


@pytest.mark.parametrize('strategy', [0, 1])
def test_minuit_strategy_global(mocker, backend, strategy):
    pyhf.set_backend(
        pyhf.tensorlib, pyhf.optimize.minuit_optimizer(strategy=strategy, tolerance=0.2)
    )
    spy = mocker.spy(pyhf.optimize.minuit_optimizer, '_minimize')
    m = pyhf.simplemodels.uncorrelated_background([50.0], [100.0], [10.0])
    data = pyhf.tensorlib.astensor([125.0] + m.config.auxdata)

    do_grad = pyhf.tensorlib.default_do_grad
    pyhf.infer.mle.fit(data, m)
    assert spy.call_count == 1
    assert spy.spy_return.minuit.strategy == strategy if do_grad else 1

    pyhf.infer.mle.fit(data, m, strategy=0)
    assert spy.call_count == 2
    assert spy.spy_return.minuit.strategy == 0

    pyhf.infer.mle.fit(data, m, strategy=1)
    assert spy.call_count == 3
    assert spy.spy_return.minuit.strategy == 1


def test_set_tolerance(backend):
    m = pyhf.simplemodels.uncorrelated_background([50.0], [100.0], [10.0])
    data = pyhf.tensorlib.astensor([125.0] + m.config.auxdata)

    assert pyhf.infer.mle.fit(data, m, tolerance=0.01) is not None

    pyhf.set_backend(pyhf.tensorlib, pyhf.optimize.scipy_optimizer(tolerance=0.01))
    assert pyhf.infer.mle.fit(data, m) is not None

    pyhf.set_backend(pyhf.tensorlib, pyhf.optimize.minuit_optimizer(tolerance=0.01))
    assert pyhf.infer.mle.fit(data, m) is not None


@pytest.mark.parametrize(
    'optimizer',
    [pyhf.optimize.scipy_optimizer, pyhf.optimize.minuit_optimizer],
    ids=['scipy', 'minuit'],
)
def test_optimizer_unsupported_minimizer_options(optimizer):
    pyhf.set_backend(pyhf.default_backend, optimizer())
    m = pyhf.simplemodels.uncorrelated_background([5.0], [10.0], [3.5])
    data = pyhf.tensorlib.astensor([10.0] + m.config.auxdata)
    with pytest.raises(pyhf.exceptions.Unsupported) as excinfo:
        pyhf.infer.mle.fit(data, m, unsupported_minimizer_options=False)
    assert 'unsupported_minimizer_options' in str(excinfo.value)


@pytest.mark.parametrize('return_result_obj', [False, True], ids=['no_obj', 'obj'])
@pytest.mark.parametrize('return_fitted_val', [False, True], ids=['no_fval', 'fval'])
@pytest.mark.parametrize(
    'optimizer',
    [pyhf.optimize.scipy_optimizer, pyhf.optimize.minuit_optimizer],
    ids=['scipy', 'minuit'],
)
def test_optimizer_return_values(optimizer, return_fitted_val, return_result_obj):
    pyhf.set_backend(pyhf.default_backend, optimizer())
    m = pyhf.simplemodels.uncorrelated_background([5.0], [10.0], [3.5])
    data = pyhf.tensorlib.astensor([10.0] + m.config.auxdata)
    result = pyhf.infer.mle.fit(
        data,
        m,
        return_fitted_val=return_fitted_val,
        return_result_obj=return_result_obj,
    )

    if not return_fitted_val and not return_result_obj:
        assert not isinstance(result, tuple)
        assert len(result) == 2
    else:
        assert isinstance(result, tuple)
        assert len(result) == sum([1, return_fitted_val, return_result_obj])

    if return_result_obj:
        assert isinstance(result[-1], OptimizeResult)


@pytest.fixture(scope='module')
def source():
    source = {
        'binning': [2, -0.5, 1.5],
        'bindata': {
            'data': [120.0, 180.0],
            'bkg': [100.0, 150.0],
            'bkgsys_up': [102, 190],
            'bkgsys_dn': [98, 100],
            'sig': [30.0, 95.0],
        },
    }
    return source


@pytest.fixture(scope='module')
def spec(source):
    spec = {
        'channels': [
            {
                'name': 'singlechannel',
                'samples': [
                    {
                        'name': 'signal',
                        'data': source['bindata']['sig'],
                        'modifiers': [
                            {'name': 'mu', 'type': 'normfactor', 'data': None}
                        ],
                    },
                    {
                        'name': 'background',
                        'data': source['bindata']['bkg'],
                        'modifiers': [
                            {
                                'name': 'bkg_norm',
                                'type': 'histosys',
                                'data': {
                                    'lo_data': source['bindata']['bkgsys_dn'],
                                    'hi_data': source['bindata']['bkgsys_up'],
                                },
                            }
                        ],
                    },
                ],
            }
        ]
    }
    return spec


@pytest.mark.parametrize('mu', [1.0], ids=['mu=1'])
def test_optim(backend, source, spec, mu):
    pdf = pyhf.Model(spec)
    data = source['bindata']['data'] + pdf.config.auxdata

    init_pars = pdf.config.suggested_init()
    par_bounds = pdf.config.suggested_bounds()

    optim = pyhf.optimizer

    result = optim.minimize(pyhf.infer.mle.twice_nll, data, pdf, init_pars, par_bounds)
    assert pyhf.tensorlib.tolist(result)

    result = optim.minimize(
        pyhf.infer.mle.twice_nll,
        data,
        pdf,
        init_pars,
        par_bounds,
        fixed_vals=[(pdf.config.poi_index, mu)],
    )
    assert pyhf.tensorlib.tolist(result)


@pytest.mark.parametrize('mu', [1.0], ids=['mu=1'])
def test_optim_with_value(backend, source, spec, mu):
    pdf = pyhf.Model(spec)
    data = source['bindata']['data'] + pdf.config.auxdata

    init_pars = pdf.config.suggested_init()
    par_bounds = pdf.config.suggested_bounds()

    optim = pyhf.optimizer

    result = optim.minimize(pyhf.infer.mle.twice_nll, data, pdf, init_pars, par_bounds)
    assert pyhf.tensorlib.tolist(result)

    result, fitted_val = optim.minimize(
        pyhf.infer.mle.twice_nll,
        data,
        pdf,
        init_pars,
        par_bounds,
        fixed_vals=[(pdf.config.poi_index, mu)],
        return_fitted_val=True,
    )
    assert pyhf.tensorlib.tolist(result)
    assert pyhf.tensorlib.shape(fitted_val) == ()
    assert pytest.approx(17.52954975, rel=1e-5) == pyhf.tensorlib.tolist(fitted_val)


@pytest.mark.parametrize('mu', [1.0], ids=['mu=1'])
@pytest.mark.only_numpy_minuit
def test_optim_uncerts(backend, source, spec, mu):
    pdf = pyhf.Model(spec)
    data = source['bindata']['data'] + pdf.config.auxdata

    init_pars = pdf.config.suggested_init()
    par_bounds = pdf.config.suggested_bounds()

    optim = pyhf.optimizer

    result = optim.minimize(pyhf.infer.mle.twice_nll, data, pdf, init_pars, par_bounds)
    assert pyhf.tensorlib.tolist(result)

    result = optim.minimize(
        pyhf.infer.mle.twice_nll,
        data,
        pdf,
        init_pars,
        par_bounds,
        fixed_vals=[(pdf.config.poi_index, mu)],
        return_uncertainties=True,
    )
    assert result.shape == (2, 2)
    assert pytest.approx([0.26418431, 0.0]) == pyhf.tensorlib.tolist(result[:, 1])


@pytest.mark.parametrize('mu', [1.0], ids=['mu=1'])
@pytest.mark.only_numpy_minuit
def test_optim_correlations(backend, source, spec, mu):
    pdf = pyhf.Model(spec)
    data = source['bindata']['data'] + pdf.config.auxdata

    init_pars = pdf.config.suggested_init()
    par_bounds = pdf.config.suggested_bounds()

    optim = pyhf.optimizer

    result = optim.minimize(pyhf.infer.mle.twice_nll, data, pdf, init_pars, par_bounds)
    assert pyhf.tensorlib.tolist(result)

    result, correlations = optim.minimize(
        pyhf.infer.mle.twice_nll,
        data,
        pdf,
        init_pars,
        par_bounds,
        [(pdf.config.poi_index, mu)],
        return_correlations=True,
    )
    assert result.shape == (2,)
    assert correlations.shape == (2, 2)
    assert pyhf.tensorlib.tolist(result)
    assert pyhf.tensorlib.tolist(correlations)

    assert np.allclose([[1.0, 0.0], [0.0, 0.0]], pyhf.tensorlib.tolist(correlations))


@pytest.mark.parametrize(
    'has_reached_call_limit', [False, True], ids=['no_call_limit', 'call_limit']
)
@pytest.mark.parametrize(
    'is_above_max_edm', [False, True], ids=['below_max_edm', 'above_max_edm']
)
def test_minuit_failed_optimization(
    monkeypatch, mocker, has_reached_call_limit, is_above_max_edm
):
    class BadMinuit(iminuit.Minuit):
        @property
        def valid(self):
            return False

        @property
        def fmin(self):
            mock = mocker.MagicMock()
            mock.has_reached_call_limit = has_reached_call_limit
            mock.is_above_max_edm = is_above_max_edm
            return mock

    monkeypatch.setattr(iminuit, 'Minuit', BadMinuit)
    pyhf.set_backend('numpy', 'minuit')
    pdf = pyhf.simplemodels.uncorrelated_background([5], [10], [3.5])
    data = [10] + pdf.config.auxdata
    spy = mocker.spy(pyhf.optimize.minuit_optimizer, '_minimize')
    with pytest.raises(pyhf.exceptions.FailedMinimization) as excinfo:
        pyhf.infer.mle.fit(data, pdf)

    assert isinstance(excinfo.value.result, OptimizeResult)

    assert excinfo.match('Optimization failed')
    assert 'Optimization failed' in spy.spy_return.message
    if has_reached_call_limit:
        assert excinfo.match('Call limit was reached')
        assert 'Call limit was reached' in spy.spy_return.message
    if is_above_max_edm:
        assert excinfo.match('Estimated distance to minimum too large')
        assert 'Estimated distance to minimum too large' in spy.spy_return.message


def test_minuit_set_options(mocker):
    pyhf.set_backend('numpy', 'minuit')
    pdf = pyhf.simplemodels.uncorrelated_background([5], [10], [3.5])
    data = [10] + pdf.config.auxdata
    # no need to postprocess in this test
    mocker.patch.object(OptimizerMixin, '_internal_postprocess')
    spy = mocker.spy(pyhf.optimize.minuit_optimizer, '_minimize')
    pyhf.infer.mle.fit(data, pdf, tolerance=0.5, strategy=0)
    assert spy.spy_return.minuit.tol == 0.5
    assert spy.spy_return.minuit.strategy == 0


def test_get_tensor_shim(monkeypatch):
    monkeypatch.setattr(pyhf.tensorlib, 'name', 'fake_backend')
    with pytest.raises(ValueError) as excinfo:
        _get_tensor_shim()

    assert 'No optimizer shim for fake_backend.' == str(excinfo.value)


def test_stitch_pars(backend):
    tb, _ = backend

    passthrough = _make_stitch_pars()
    pars = ['a', 'b', 1.0, 2.0, object()]
    assert passthrough(pars) == pars

    fixed_idx = [0, 3, 4]
    variable_idx = [1, 2, 5]
    fixed_vals = [10, 40, 50]
    variable_vals = [20, 30, 60]
    tv = _TensorViewer([fixed_idx, variable_idx])
    stitch_pars = _make_stitch_pars(tv, fixed_vals)

    pars = tb.astensor(variable_vals)
    assert tb.tolist(stitch_pars(pars)) == [10, 20, 30, 40, 50, 60]
    assert tb.tolist(stitch_pars(pars, stitch_with=tb.zeros(3))) == [
        0,
        20,
        30,
        0,
        0,
        60,
    ]


def test_init_pars_sync_fixed_values_scipy(mocker):
    opt = pyhf.optimize.scipy_optimizer()

    minimizer = mocker.MagicMock()
    opt._minimize(minimizer, None, [9, 9, 9], fixed_vals=[(0, 1)])
    assert minimizer.call_args[0] == (None, [1, 9, 9])


def test_init_pars_sync_fixed_values_minuit(mocker):
    opt = pyhf.optimize.minuit_optimizer()

    # patch all we need
    from pyhf.optimize import opt_minuit

    minuit = mocker.patch.object(getattr(opt_minuit, 'iminuit'), 'Minuit')
    minimizer = opt._get_minimizer(None, [9, 9, 9], [(0, 10)] * 3, fixed_vals=[(0, 1)])
    assert minuit.called
    # python 3.7 does not have ::args attribute on ::call_args
    # assert minuit.call_args.args[1] == [1, 9, 9]
    assert minuit.call_args[0][1] == [1, 9, 9]
    assert minimizer.fixed == [True, False, False]


def test_step_sizes_fixed_parameters_minuit(mocker):
    opt = pyhf.optimize.minuit_optimizer()

    # patch all we need
    from pyhf.optimize import opt_minuit

    minuit = mocker.patch.object(getattr(opt_minuit, 'iminuit'), 'Minuit')
    minimizer = opt._get_minimizer(None, [9, 9, 9], [(0, 10)] * 3, fixed_vals=[(0, 1)])

    assert minuit.called
    assert minimizer.fixed == [True, False, False]
    assert minimizer.errors == [0.0, 0.01, 0.01]


def test_solver_options_behavior_scipy(mocker):
    opt = pyhf.optimize.scipy_optimizer(solver_options={'arbitrary_option': 'foobar'})

    minimizer = mocker.MagicMock()
    opt._minimize(minimizer, None, [9, 9, 9], fixed_vals=[(0, 1)])
    assert 'arbitrary_option' in minimizer.call_args[1]['options']
    assert minimizer.call_args[1]['options']['arbitrary_option'] == 'foobar'

    opt._minimize(
        minimizer,
        None,
        [9, 9, 9],
        fixed_vals=[(0, 1)],
        options={'solver_options': {'arbitrary_option': 'baz'}},
    )
    assert 'arbitrary_option' in minimizer.call_args[1]['options']
    assert minimizer.call_args[1]['options']['arbitrary_option'] == 'baz'


def test_solver_options_scipy(mocker):
    optimizer = pyhf.optimize.scipy_optimizer(solver_options={'ftol': 1e-5})
    pyhf.set_backend('numpy', optimizer)
    assert pyhf.optimizer.solver_options == {'ftol': 1e-5}

    model = pyhf.simplemodels.uncorrelated_background([50.0], [100.0], [10.0])
    data = pyhf.tensorlib.astensor([125.0] + model.config.auxdata)
    assert pyhf.infer.mle.fit(data, model).tolist()


# Note: in this case, scipy won't usually raise errors for arbitrary options
# so this test exists as a sanity reminder that scipy is not perfect.
# It does raise a scipy.optimize.OptimizeWarning though.
def test_bad_solver_options_scipy(mocker):
    optimizer = pyhf.optimize.scipy_optimizer(
        solver_options={'arbitrary_option': 'foobar'}
    )
    pyhf.set_backend('numpy', optimizer)
    assert pyhf.optimizer.solver_options == {'arbitrary_option': 'foobar'}

    model = pyhf.simplemodels.uncorrelated_background([50.0], [100.0], [10.0])
    data = pyhf.tensorlib.astensor([125.0] + model.config.auxdata)

    with pytest.warns(
        OptimizeWarning, match="Unknown solver options: arbitrary_option"
    ):
        assert pyhf.infer.mle.fit(data, model).tolist()


def test_minuit_param_names(mocker):
    pyhf.set_backend('numpy', 'minuit')
    pdf = pyhf.simplemodels.uncorrelated_background([5], [10], [3.5])
    data = [10] + pdf.config.auxdata
    _, result = pyhf.infer.mle.fit(data, pdf, return_result_obj=True)
    assert 'minuit' in result
    assert result.minuit.parameters == ('mu', 'uncorr_bkguncrt[0]')

    pdf.config.par_names = mocker.Mock(return_value=None)
    _, result = pyhf.infer.mle.fit(data, pdf, return_result_obj=True)
    assert 'minuit' in result
    assert result.minuit.parameters == ('x0', 'x1')
