import pyhf
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


# bins = [1, 10, 50, 100, 200, 500, 800, 1000]
bins = [50, 500]
bin_ids = ['{}_bins'.format(n_bins) for n_bins in bins]


@pytest.mark.parametrize('n_bins', bins, ids=bin_ids)
def test_runOnePoint_q_mu(n_bins,
                          tolerance={
                              'numpy': 1e-02,
                              'tensors': 5e-03
                          }):
    """
    Check that the different backends all compute a test statistic
    that is within a specific tolerance of each other.

    Args:
        n_bins: `list` of number of bins given by pytest parameterization
        tolerance: `dict` of the maximum differences the test statistics
                    can differ relative to each other

    Returns:
        None
    """

    source = generate_source_static(n_bins)
    pdf = hepdata_like(source['bindata']['sig'],
                       source['bindata']['bkg'],
                       source['bindata']['bkgerr'])
    data = source['bindata']['data'] + pdf.config.auxdata

    backends = [
        pyhf.tensor.numpy_backend(poisson_from_normal=True),
        pyhf.tensor.tensorflow_backend(session=tf.Session()),
        pyhf.tensor.pytorch_backend(),
        # mxnet_backend()
    ]

    test_statistic = []
    for backend in backends:
        pyhf.set_backend(backend)

        if isinstance(pyhf.tensorlib, pyhf.tensor.tensorflow_backend):
            tf.reset_default_graph()
            pyhf.tensorlib.session = tf.Session()

        q_mu = pyhf.utils.runOnePoint(1.0, data, pdf,
                                      pdf.config.suggested_init(),
                                      pdf.config.suggested_bounds())[0]
        test_statistic.append(pyhf.tensorlib.tolist(q_mu))

    # compare to NumPy/SciPy
    test_statistic = np.array(test_statistic)
    numpy_ratio = np.divide(test_statistic, test_statistic[0])
    numpy_ratio_delta_unity = np.absolute(np.subtract(numpy_ratio, 1))

    # compare tensor libraries to each other
    tensors_ratio = np.divide(test_statistic[1], test_statistic[2])
    tensors_ratio_delta_unity = np.absolute(np.subtract(tensors_ratio, 1))

    try:
        assert (numpy_ratio_delta_unity < tolerance['numpy']).all()
    except AssertionError:
        print('Ratio to NumPy+SciPy exceeded tolerance of {}: {}'.format(
            tolerance['numpy'], numpy_ratio_delta_unity.tolist()))
        assert False
    try:
        assert (tensors_ratio_delta_unity < tolerance['tensors']).all()
    except AssertionError:
        print('Ratio between tensor backends exceeded tolerance of {}: {}'.format(
            tolerance['tensors'], tensors_ratio_delta_unity.tolist()))
        assert False
