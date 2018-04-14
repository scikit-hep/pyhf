import pyhf
import tensorflow as tf
import pytest


@pytest.fixture
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


@pytest.fixture
def spec(source):
    spec = {
        'channels': [{
            'name': 'singlechannel',
            'samples': [{
                        'name': 'signal',
                        'data': source['bindata']['sig'],
                        'modifiers': [{
                                'name': 'mu',
                                'type': 'normfactor',
                                'data': None
                        }]},
                        {
                        'name': 'background',
                        'data': source['bindata']['bkg'],
                        'modifiers': [{
                                'name': 'bkg_norm',
                                'type': 'histosys',
                                'data': {
                                    'lo_data': source['bindata']['bkgsys_dn'],
                                    'hi_data': source['bindata']['bkgsys_up']
                                }
                        }]}]
        }]
    }
    return spec


@pytest.mark.parametrize('backend',
                         [
                             numpy_backend(poisson_from_normal=True),
                             tensorflow_backend(session=tf.Session()),
                             pytorch_backend(poisson_from_normal=True),
                             # mxnet_backend(),
                         ],
                         ids=[
                             'numpy',
                             'tensorflow',
                             'pytorch',
                             # 'mxnet',
                         ])
def test_optim(source, spec, backend):
    pdf = pyhf.hfpdf(spec)
    data = source['bindata']['data'] + pdf.config.auxdata

    init_pars = pdf.config.suggested_init()
    par_bounds = pdf.config.suggested_bounds()

    pyhf.set_backend(pyhf.tensor.tensorflow_backend())
    pyhf.tensorlib.session = tf.Session()
    optim = pyhf.optimizer

    result = optim.unconstrained_bestfit(
        pyhf.loglambdav, data, pdf, init_pars, par_bounds)
    try:
        assert pyhf.tensorlib.tolist(result)
    except AssertionError:
        print('unconstrained_bestfit failed')
        pyhf.set_backend(oldlib)
        assert False

    result = optim.constrained_bestfit(
        pyhf.loglambdav, 1.0, data, pdf, init_pars, par_bounds)
    try:
        assert pyhf.tensorlib.tolist(result)
    except AssertionError:
        print('constrained_bestfit failed')
        pyhf.set_backend(oldlib)
        assert False
