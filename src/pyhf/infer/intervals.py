"""Interval estimation"""
from pyhf.infer import hypotest
from pyhf import get_backend
import numpy as np

__all__ = ["upperlimit"]


def __dir__():
    return __all__


def _interp(x, xp, fp):
    tb, _ = get_backend()
    return tb.astensor(np.interp(x, xp.tolist(), fp.tolist()))


def upperlimit(data, model, scan, level=0.05, return_results=False, **hypotest_kwargs):
    """
    Calculate an upper limit interval ``(0, poi_up)`` for a single
    Parameter of Interest (POI) using a fixed scan through POI-space.

    Example:
        >>> import numpy as np
        >>> import pyhf
        >>> pyhf.set_backend("numpy")
        >>> model = pyhf.simplemodels.uncorrelated_background(
        ...     signal=[12.0, 11.0], bkg=[50.0, 52.0], bkg_uncertainty=[3.0, 7.0]
        ... )
        >>> observations = [51, 48]
        >>> data = pyhf.tensorlib.astensor(observations + model.config.auxdata)
        >>> scan = np.linspace(0, 5, 21)
        >>> obs_limit, exp_limits, (scan, results) = pyhf.infer.intervals.upperlimit(
        ...     data, model, scan, return_results=True
        ... )
        >>> obs_limit
        array(1.01764089)
        >>> exp_limits
        [array(0.59576921), array(0.76169166), array(1.08504773), array(1.50170482), array(2.06654952)]

    Args:
        data (:obj:`tensor`): The observed data.
        model (~pyhf.pdf.Model): The statistical model adhering to the schema ``model.json``.
        scan (:obj:`iterable`): Iterable of POI values.
        level (:obj:`float`): The threshold value to evaluate the interpolated results at.
        return_results (:obj:`bool`): Whether to return the per-point results.
        hypotest_kwargs (:obj:`string`): Kwargs for the calls to
         :class:`~pyhf.infer.hypotest` to configure the fits.

    Returns:
        Tuple of Tensors:

            - Tensor: The observed upper limit on the POI.
            - Tensor: The expected upper limits on the POI.
            - Tuple of Tensors: The given ``scan`` along with the
              :class:`~pyhf.infer.hypotest` results at each test POI.
              Only returned when ``return_results`` is ``True``.
    """
    tb, _ = get_backend()
    results = [
        hypotest(mu, data, model, return_expected_set=True, **hypotest_kwargs)
        for mu in scan
    ]
    obs = tb.astensor([[r[0]] for r in results])
    exp = tb.astensor([[r[1][idx] for idx in range(5)] for r in results])

    result_arrary = tb.concatenate([obs, exp], axis=1).T

    # observed limit and the (0, +-1, +-2)sigma expected limits
    limits = [_interp(level, result_arrary[idx][::-1], scan[::-1]) for idx in range(6)]
    obs_limit, exp_limits = limits[0], limits[1:]

    if return_results:
        return obs_limit, exp_limits, (scan, results)
    return obs_limit, exp_limits
