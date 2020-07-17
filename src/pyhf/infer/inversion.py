from . import hypotest
from .. import get_backend


def interp(x, xp, fp):
    import numpy as np

    tb, _ = get_backend()
    return tb.astensor(np.interp(x, xp.tolist(), fp.tolist()))


def upperlimit(data, model, scan, return_results=False):
    tb, _ = get_backend()
    results = [hypotest(mu, data, model, return_expected_set=True) for mu in scan]
    obs = tb.astensor([[r[0][0]] for r in results])
    exp = tb.astensor([[r[1][i][0] for i in range(5)] for r in results])
    resarary = tb.concatenate([obs, exp], axis=1).T

    ol = interp(0.05, resarary[0][::-1], scan[::-1])
    el = [interp(0.05, resarary[i][::-1], scan[::-1]) for i in range(5)]

    if return_results:
        return ol, el, (scan, results)
    return ol, el
