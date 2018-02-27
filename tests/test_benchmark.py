import pyhf
from pyhf.tensor.pytorch_backend import pytorch_backend
from pyhf.tensor.numpy_backend import numpy_backend
from pyhf.tensor.tensorflow_backend import tensorflow_backend
from pyhf.tensor.mxnet_backend import mxnet_backend
from pyhf.simplemodels import hepdata_like
import tensorflow as tf

import pytest
tf_sess = tf.Session()


def generate_source(n_bins):
    """
    Create the source structure for the given number of bins.
    Currently the data that are being put in is all the same
    but in the future it should be generated pseudodata.
    """
    binning = [n_bins, -0.5, 1.5]
    data = [120.0]
    bkg = [100.0]
    bkgerr = [10.0]
    sig = [30.0]
    if n_bins > 1:
        for bin in range(1, n_bins):
            data.append(180.0)
            bkg.append(150.0)
            bkgerr.append(10.0)
            sig.append(95.0)
    source = {
        'binning': binning,
        'bindata': {
            'data': data,
            'bkg': bkg,
            'bkgerr': bkgerr,
            'sig': sig
        }
    }
    return source


def select_backend(backend):
    if backend.lower() == 'numpy':
        return numpy_backend(poisson_from_normal=True)
    elif backend.lower() == 'tensorflow':
        return tensorflow_backend(session=tf_sess)
    elif backend.lower() == 'pytorch':
        return pytorch_backend()
    elif backend.lower() == 'mxnet':
        return mxnet_backend()


def logpdf(source):
    pdf = hepdata_like(source['bindata']['sig'],
                       source['bindata']['bkg'],
                       source['bindata']['bkgerr'])
    data = source['bindata']['data'] + pdf.config.auxdata

    return pdf.logpdf(pdf.config.suggested_init(), data)


# At the moment the backends aren't doing anything, but are POC
@pytest.mark.parametrize('n_bins', [1, 2, 3, 4, 5, 10, 15, 20, 25, 50, 100])
@pytest.mark.parametrize('backend', ['numpy', 'tensorflow', 'pytorch', 'mxnet'])
def test_logpdf(benchmark, backend, n_bins):
    """
    Benchmark the performance of logpdf for various numbers of bins
    """
    pyhf.tensorlib = select_backend(backend)
    source = generate_source(n_bins)
    assert benchmark(logpdf, source) is not None
