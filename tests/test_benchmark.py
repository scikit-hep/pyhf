import pyhf
from pyhf.tensor.numpy_backend import numpy_backend
from pyhf.tensor.tensorflow_backend import tensorflow_backend
from pyhf.tensor.pytorch_backend import pytorch_backend
from pyhf.tensor.mxnet_backend import mxnet_backend
from pyhf.simplemodels import hepdata_like
import tensorflow as tf
import numpy as np
import pytest


def generate_source_static(n_bins):
    """
    Create the source structure for the given number of bins.

    Args:
        n_bins: `list` of number of bins

    Returns:
        source
    """
    binning = [n_bins, -0.5, n_bins + 0.5]
    data = [120.0] * n_bins
    bkg = [100.0] * n_bins
    bkgerr = [10.0] * n_bins
    sig = [30.0] * n_bins

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


def generate_source_poisson(n_bins):
    """
    Create the source structure for the given number of bins.
    Sample from a Poisson distribution

    Args:
        n_bins: `list` of number of bins

    Returns:
        source
    """
    np.random.seed(0)  # Fix seed for reproducibility
    binning = [n_bins, -0.5, n_bins + 0.5]
    data = np.random.poisson(120.0, n_bins).tolist()
    bkg = np.random.poisson(100.0, n_bins).tolist()
    bkgerr = np.random.poisson(10.0, n_bins).tolist()
    sig = np.random.poisson(30.0, n_bins).tolist()

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


def runOnePoint(pdf, data):
    result = pyhf.runOnePoint(1.0, data, pdf,
                              pdf.config.suggested_init(),
                              pdf.config.suggested_bounds())

    # Reset the TensorFlow graph and session for each run
    if isinstance(pyhf.tensorlib, tensorflow_backend):
        tf.reset_default_graph()
        pyhf.tensorlib.session = tf.Session()

    return result


# bins = [1, 10, 50, 100, 200, 500, 800, 1000]
bins = [10, 100, 200]
bin_ids = ['{}_bins'.format(n_bins) for n_bins in bins]


@pytest.mark.parametrize('n_bins', bins, ids=bin_ids)
@pytest.mark.parametrize('backend',
                         [
                             numpy_backend(poisson_from_normal=True),
                             tensorflow_backend(session=tf.Session()),
                             pytorch_backend(),
                             # mxnet_backend(),
                         ],
                         ids=[
                             'numpy',
                             'tensorflow',
                             'pytorch',
                             # 'mxnet',
                         ])
def test_runOnePoint(benchmark, backend, n_bins):
    """
    Benchmark the performance of pyhf.runOnePoint()
    for various numbers of bins and different backends

    Args:
        benchmark: pytest benchmark
        backend: `pyhf` tensorlib given by pytest parameterization
        n_bins: `list` of number of bins given by pytest parameterization

    Returns:
        None
    """
    default_backend = pyhf.tensorlib
    pyhf.set_backend(backend)

    source = generate_source_static(n_bins)
    pdf = hepdata_like(source['bindata']['sig'],
                       source['bindata']['bkg'],
                       source['bindata']['bkgerr'])
    data = source['bindata']['data'] + pdf.config.auxdata
    assert benchmark(runOnePoint, pdf, data) is not None

    # Reset backend
    pyhf.set_backend(default_backend)
