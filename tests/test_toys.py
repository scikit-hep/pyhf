import pyhf
import numpy as np


def test_smoketest_toys(backend):
    tb, _ = backend
    m = pyhf.simplemodels.hepdata_like([6], [9], [3])
    s = m.make_pdf(pyhf.tensorlib.astensor(m.config.suggested_init()))
    assert np.asarray(tb.tolist(s.log_prob(s.sample((1000,))))).shape == (1000,)
