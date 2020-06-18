import pyhf
import pytest
from scipy.optimize import minimize


def test_get_invalid_optimizer():
    with pytest.raises(pyhf.exceptions.InvalidOptimizer):
        assert pyhf.optimize.scipy


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
