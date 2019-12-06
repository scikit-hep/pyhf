import pyhf
import numpy as np


def test_smoketest_toys(backend):
    tb, _ = backend
    m = pyhf.simplemodels.hepdata_like([6], [9], [3])
    s = m.make_pdf(pyhf.tensorlib.astensor(m.config.suggested_init()))
    assert np.asarray(tb.tolist(s.log_prob(s.sample((1000,))))).shape == (1000,)

    tb, _ = backend
    m = pyhf.simplemodels.hepdata_like([6, 6], [9, 9], [3, 3], batch_size=13)
    s = m.make_pdf(pyhf.tensorlib.astensor(m.batch_size * [m.config.suggested_init()]))
    assert np.asarray(tb.tolist(s.sample((10,)))).shape == (10, 13, 4)
