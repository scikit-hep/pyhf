import pyhf
import pytest


def test_get_invalid_optimizer():
    with pytest.raises(pyhf.exceptions.InvalidOptimizer):
        assert pyhf.optimize.scipy


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
@pytest.mark.skip_mxnet
def test_optim(backend, source, spec, mu):
    pdf = pyhf.Model(spec)
    data = source['bindata']['data'] + pdf.config.auxdata

    init_pars = pdf.config.suggested_init()
    par_bounds = pdf.config.suggested_bounds()

    optim = pyhf.optimizer

    result = optim.unconstrained_bestfit(
        pyhf.utils.loglambdav, data, pdf, init_pars, par_bounds
    )
    assert pyhf.tensorlib.tolist(result)

    result = optim.constrained_bestfit(
        pyhf.utils.loglambdav, mu, data, pdf, init_pars, par_bounds
    )
    assert pyhf.tensorlib.tolist(result)
