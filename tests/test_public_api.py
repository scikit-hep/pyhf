import pyhf
import numpy as np


def test_smoketest(backend):
    np.random.seed(0)
    n_bins = 500
    model = pyhf.simplemodels.hepdata_like([10] * n_bins, [50] * n_bins, [1] * n_bins)
    parameters = model.config.suggested_init()
    data = np.random.randint(50, 60, size=n_bins).tolist() + model.config.auxdata
    model.logpdf(parameters, data)
