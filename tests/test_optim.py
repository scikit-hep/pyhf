import pyhf
import tensorflow as tf
import pytest


@pytest.fixture(scope='module')
def source():
    source = {
        'binning': [2, -0.5, 1.5],
        'bindata': {
            'data': [120.0, 180.0],
            'bkg': [100.0, 150.0],
            'bkgsys_up': [102, 190],
            'bkgsys_dn': [98, 100],
            'sig': [30.0, 95.0]
        }
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
                            {
                                'name': 'mu',
                                'type': 'normfactor',
                                'data': None
                            }
                        ]
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
                                    'hi_data': source['bindata']['bkgsys_up']
                                }
                            }
                        ]
                    }
                ]
            }
        ]
    }
    return spec


@pytest.mark.parametrize('mu',
                         [
                             1.,
                         ],
                         ids=[
                             'mu=1',
                         ])
@pytest.mark.parametrize('backend',
                         [
                             pyhf.tensor.numpy_backend(poisson_from_normal=True),
                             pyhf.tensor.tensorflow_backend(session=tf.Session()),
                             pyhf.tensor.pytorch_backend(poisson_from_normal=True),
                             # pyhf.tensor.mxnet_backend(),
                         ],
                         ids=[
                             'numpy',
                             'tensorflow',
                             'pytorch',
                             # 'mxnet',
                         ])
def test_optim(source, spec, mu, backend):
    pdf = pyhf.Model(spec)
    data = source['bindata']['data'] + pdf.config.auxdata

    init_pars = pdf.config.suggested_init()
    par_bounds = pdf.config.suggested_bounds()

    pyhf.set_backend(backend)
    optim = pyhf.optimizer
    if isinstance(pyhf.tensorlib, pyhf.tensor.tensorflow_backend):
        tf.reset_default_graph()
        pyhf.tensorlib.session = tf.Session()

    result = optim.unconstrained_bestfit(
        pyhf.utils.loglambdav, data, pdf, init_pars, par_bounds)
    assert pyhf.tensorlib.tolist(result)

    result = optim.constrained_bestfit(
        pyhf.utils.loglambdav, mu, data, pdf, init_pars, par_bounds)
    assert pyhf.tensorlib.tolist(result)
