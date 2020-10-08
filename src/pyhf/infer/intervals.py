from . import hypotest
from .. import get_backend
import numpy as np


def _interp(x, xp, fp):
    tb, _ = get_backend()
    return tb.astensor(np.interp(x, xp.tolist(), fp.tolist()))


def upperlimit(data, model, scan, level=0.05, return_results=False):
    """
    Calculate an upper limit interval ``(0, poi_up)`` for a single
    Parameter of Interest (POI) using a fixed scan through POI-space.

    Args:
        data (`tensor`): The observed data.
        model (~pyhf.pdf.Model): The statistical model adhering to the schema ``model.json``.
        scan (`Iterable`): Iterable of POI values.
        level (`float`): The threshold value to evaluate the interpolated results at.
        return_results (`bool`): Whether to return the per-point results.
                                 Default is ``False``.

    Returns:
        Tuple of Tensors:

            - Tensor: The observed upper limit on the POI.
            - Tensor: The expected upper limit on the POI.
            - Tuple of Tensors: The given ``scan`` along with the
              :class:`~pyhf.infer.hypotest` results at each test POI.
              Only returned when ``return_results`` is ``True``.
    """
    tb, _ = get_backend()
    results = [hypotest(mu, data, model, return_expected_set=True) for mu in scan]
    obs = tb.astensor([[r[0][0]] for r in results])
    exp = tb.astensor([[r[1][i][0] for i in range(5)] for r in results])
    resarary = tb.concatenate([obs, exp], axis=1).T

    limits = [_interp(level, resarary[i][::-1], scan[::-1]) for i in range(6)]

    if return_results:
        return limits[0], limits[1:], (scan, results)
    return limits[0], limits[1:]
