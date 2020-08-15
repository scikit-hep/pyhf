from . import hypotest
from .. import get_backend


def _interp(x, xp, fp):
    import numpy as np

    tb, _ = get_backend()
    return tb.astensor(np.interp(x, xp.tolist(), fp.tolist()))


def upperlimit(data, model, scan, level=0.05, return_results=False):
    '''
    Calculate an upper limit interval (0,poi_up) for a single
    Parameter of Interest (POI) using a fixed scan through
    POI-space.

    Args:
        data (tensor): the observed data
        model (pyhf.Model): the statistical model
        scan (Iterable): iterable of poi values
        return_results (bool): whether to return the per-point results

    Returns:
        observed limit (tensor)
        expected limit (tensor)
        scan results (tuple  (tensor, tensor))
    '''
    tb, _ = get_backend()
    results = [hypotest(mu, data, model, return_expected_set=True) for mu in scan]
    obs = tb.astensor([[r[0][0]] for r in results])
    exp = tb.astensor([[r[1][i][0] for i in range(5)] for r in results])
    resarary = tb.concatenate([obs, exp], axis=1).T

    limits = [_interp(level, resarary[i][::-1], scan[::-1]) for i in range(6)]

    if return_results:
        return limits[0], limits[1:], (scan, results)
    return limits[0], limits[1:]
