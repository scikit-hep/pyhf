import pyhf
from pyhf.optimize.mixins import OptimizerMixin
from pyhf.optimize.common import _get_tensor_shim
import pytest
from scipy.optimize import minimize
import iminuit


# from https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html#nelder-mead-simplex-algorithm-method-nelder-mead
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


@pytest.mark.parametrize(
    'precision', ['32b', '64b'], ids=['32b', '64b'],
)
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
def test_minimize(tensorlib, precision, optimizer, do_grad):
    pyhf.set_backend(tensorlib(precision=precision), optimizer(grad=do_grad))
    m = pyhf.simplemodels.hepdata_like([5.0], [10.0], [3.5])
    data = pyhf.tensorlib.astensor([10.0] + m.config.auxdata)
    # numpy does not support grad
    if pyhf.tensorlib.name == 'numpy' and do_grad:
        with pytest.raises(AssertionError):
            pyhf.infer.mle.fit(data, m)
    else:
        identifier = f'{"do_grad" if pyhf.optimizer.grad else "no_grad"}-{pyhf.optimizer.name}-{pyhf.tensorlib.name}-{pyhf.tensorlib.precision}'
        expected = {
            # numpy does not do grad
            'do_grad-scipy-numpy-32b': None,
            'do_grad-scipy-numpy-64b': None,
            'do_grad-minuit-numpy-32b': None,
            'do_grad-minuit-numpy-64b': None,
            # no grad, scipy, 32b - never works
            'no_grad-scipy-numpy-32b': [1.0, 1.0],
            'no_grad-scipy-pytorch-32b': [1.0, 1.0],
            'no_grad-scipy-tensorflow-32b': [1.0, 1.0],
            'no_grad-scipy-jax-32b': [1.0, 1.0],
            # no grad, scipy, 64b - mostly consistent (~2.75e-06)
            'no_grad-scipy-numpy-64b': [2.67369062e-06, 9.99985555e-01],
            'no_grad-scipy-pytorch-64b': [2.67369062e-06, 9.99985555e-01],
            'no_grad-scipy-tensorflow-64b': [2.85259486e-06, 9.99985487e-01],
            'no_grad-scipy-jax-64b': [2.87242218e-06, 9.99985478e-01],
            # do grad, scipy, 32b - not very consistent for tensorflow
            'do_grad-scipy-pytorch-32b': [2.85133024e-06, 9.99985516e-01],
            'do_grad-scipy-tensorflow-32b': [3.46792535e-06, 9.99985337e-01],
            'do_grad-scipy-jax-32b': [2.91515607e-06, 9.99985516e-01],
            # do grad, scipy, 64b - mostly consistent (~3.0e-06)
            'do_grad-scipy-pytorch-64b': [3.00311835e-06, 9.99985456e-01],
            'do_grad-scipy-tensorflow-64b': [3.02640865e-06, 9.99985445e-01],
            'do_grad-scipy-jax-64b': [3.00311835e-06, 9.99985456e-01],
            # no grad, minuit, 32b - not very consistent
            'no_grad-minuit-numpy-32b': [1.27911707e-02, 9.95959699e-01],
            #     nb: macos gives different numerics than ubuntu for minuit pytorch 32b
            #'no_grad-minuit-pytorch-32b': [2.69202143e-02, 9.92773652e-01],
            'no_grad-minuit-pytorch-32b': [2.68813074e-02, 9.93943393e-01],
            'no_grad-minuit-tensorflow-32b': [5.47232048e-04, 9.99859154e-01],
            'no_grad-minuit-jax-32b': [8.74461184e-05, 1.0001129e00],
            #     nb: macos gives different numerics than ubuntu for minuit jax 32b
            #'no_grad-minuit-jax-32b': [1.01476780e-03, 9.9928200e-01],
            # no grad, minuit, 64b - quite consistent
            'no_grad-minuit-numpy-64b': [9.19623487e-03, 9.98248083e-01],
            'no_grad-minuit-pytorch-64b': [9.19623487e-03, 9.98248083e-01],
            'no_grad-minuit-tensorflow-64b': [9.19624519e-03, 9.98248076e-01],
            'no_grad-minuit-jax-64b': [9.19623486e-03, 9.9824808e-01],
            # do grad, minuit, 32b - not very consistent
            #     nb: macos gives different numerics than ubuntu for minuit pytorch 32b
            #'do_grad-minuit-pytorch-32b': [2.69202143e-02, 9.92773652e-01],
            'do_grad-minuit-pytorch-32b': [2.68813074e-02, 9.93943393e-01],
            'do_grad-minuit-tensorflow-32b': [5.47232048e-04, 9.99859154e-01],
            'do_grad-minuit-jax-32b': [8.74461184e-05, 1.00011289e00],
            #     nb: macos gives different numerics than ubuntu for minuit jax 32b
            #'do_grad-minuit-jax-32b': [1.01476780e-03, 9.99282002e-01],
            # do grad, minuit, 64b - quite consistent
            'do_grad-minuit-pytorch-64b': [9.19623487e-03, 9.98248083e-01],
            'do_grad-minuit-tensorflow-64b': [9.19624519e-03, 9.98248076e-01],
            'do_grad-minuit-jax-64b': [9.19623486e-03, 9.98248083e-01],
        }[identifier]

        result = pyhf.infer.mle.fit(data, m)

        atol = 1e-6
        rtol = 1e-6
        # handle cases where macos and ubuntu provide very different results numerical
        if 'minuit-pytorch-32b' in identifier:
            # not a very large difference, so we bump the relative difference down
            rtol = 2e-3
        if 'minuit-jax-32b' in identifier:
            # quite a large difference, so we bump the absolute tolerance down
            atol = 1e-3

        # check fitted parameters
        assert pytest.approx(expected, rel=rtol, abs=atol) == pyhf.tensorlib.tolist(
            result
        )


@pytest.mark.parametrize(
    'optimizer',
    [OptimizerMixin, pyhf.optimize.scipy_optimizer, pyhf.optimize.minuit_optimizer],
    ids=['mixin', 'scipy', 'minuit'],
)
def test_optimizer_mixin_extra_kwargs(optimizer):
    with pytest.raises(KeyError):
        optimizer(fake_kwarg=False)


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
        [(pdf.config.poi_index, mu)],
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
        [(pdf.config.poi_index, mu)],
        return_fitted_val=True,
    )
    assert pyhf.tensorlib.tolist(result)


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
        [(pdf.config.poi_index, mu)],
        return_uncertainties=True,
    )
    assert result.shape[1] == 2
    assert pyhf.tensorlib.tolist(result)


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
    # iminuit.Minuit = BadMinuit
    pyhf.set_backend('numpy', 'minuit')
    pdf = pyhf.simplemodels.hepdata_like([5], [10], [3.5])
    data = [10] + pdf.config.auxdata
    spy = mocker.spy(pyhf.optimize.minuit_optimizer, '_minimize')
    with pytest.raises(AssertionError):
        pyhf.infer.mle.fit(data, pdf)

    assert 'Optimization failed' in spy.spy_return.message
    if has_reached_call_limit:
        assert 'Call limit was reached' in spy.spy_return.message
    if is_above_max_edm:
        assert 'Estimated distance to minimum too large' in spy.spy_return.message


def test_get_tensor_shim(monkeypatch):
    monkeypatch.setattr(pyhf.tensorlib, 'name', 'fake_backend')
    with pytest.raises(ValueError) as excinfo:
        _get_tensor_shim()

    assert 'No optimizer shim for fake_backend.' == str(excinfo.value)
