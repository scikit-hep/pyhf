import pyhf
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
        'bindata': {'data': data, 'bkg': bkg, 'bkgerr': bkgerr, 'sig': sig},
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
        'bindata': {'data': data, 'bkg': bkg, 'bkgerr': bkgerr, 'sig': sig},
    }
    return source


# bins = [1, 10, 50, 100, 200, 500, 800, 1000]
bins = [50, 500]
bin_ids = ['{}_bins'.format(n_bins) for n_bins in bins]


@pytest.mark.parametrize('n_bins', bins, ids=bin_ids)
@pytest.mark.parametrize('invert_order', [False, True], ids=['normal', 'inverted'])
def test_hypotest_qmu_tilde(
    n_bins, invert_order, tolerance={'numpy': 1e-02, 'tensors': 5e-03}
):
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
    bkg_unc_10pc_up = [x + 0.10 * x for x in source['bindata']['bkg']]
    bkg_unc_10pc_dn = [x - 0.10 * x for x in source['bindata']['bkg']]

    signal_sample = {
        'name': 'signal',
        'data': source['bindata']['sig'],
        'modifiers': [{'name': 'mu', 'type': 'normfactor', 'data': None}],
    }

    background_sample = {
        'name': 'background',
        'data': source['bindata']['bkg'],
        'modifiers': [
            {
                'name': 'uncorr_bkguncrt',
                'type': 'shapesys',
                'data': source['bindata']['bkgerr'],
            },
            {
                'name': 'norm_bkgunc',
                'type': 'histosys',
                'hi_data': bkg_unc_10pc_up,
                'lo_data': bkg_unc_10pc_dn,
            },
        ],
    }
    samples = (
        [background_sample, signal_sample]
        if invert_order
        else [signal_sample, background_sample]
    )
    spec = {'channels': [{'name': 'singlechannel', 'samples': samples}]}
    pdf = pyhf.Model(spec)

    #   get norm_bkgunc index to set it constatnt
    norm_bkgunc_idx = -1
    for idx, par in enumerate(pdf.config.par_map):
        if par == "norm_bkgunc":
            norm_bkgunc_idx = idx
    #   Floating norm_bkgunc, fixed at nominal, plus/minus 1 sigma
    param_tests = [
        None,
        (norm_bkgunc_idx, 0.0),
        (norm_bkgunc_idx, -1.0),
        (norm_bkgunc_idx, 1.0),
    ]

    data = source['bindata']['data'] + pdf.config.auxdata

    backends = [
        pyhf.tensor.numpy_backend(precision='64b'),
        pyhf.tensor.tensorflow_backend(precision='64b'),
        pyhf.tensor.pytorch_backend(precision='64b'),
        pyhf.tensor.jax_backend(precision='64b'),
    ]

    for p in param_tests:
        fixed_vals = [p]

        test_statistic = []
        for backend in backends:
            pyhf.set_backend(backend)

            qmu_tilde = pyhf.infer.test_statistics.qmu_tilde(
                1.0,
                data,
                pdf,
                pdf.config.suggested_init(),
                pdf.config.suggested_bounds(),
                fixed_vals,
            )
            test_statistic.append(qmu_tilde)

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
            print(
                'Ratio to NumPy+SciPy exceeded tolerance of {}: {}'.format(
                    tolerance['numpy'], numpy_ratio_delta_unity.tolist()
                )
            )
            assert False
        try:
            assert (tensors_ratio_delta_unity < tolerance['tensors']).all()
        except AssertionError:
            print(
                'Ratio between tensor backends exceeded tolerance of {}: {}'.format(
                    tolerance['tensors'], tensors_ratio_delta_unity.tolist()
                )
            )
            assert False
