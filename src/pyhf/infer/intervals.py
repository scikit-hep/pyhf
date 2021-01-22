"""Interval estimation"""
from . import hypotest
from .. import get_backend
import numpy as np
from scipy.optimize import toms748 as _toms748
from warnings import warn as _warn


def _interp(x, xp, fp):
    tb, _ = get_backend()
    return tb.astensor(np.interp(x, xp.tolist(), fp.tolist()))


def upperlimit_auto(data, model, low, high, level=0.05, atol=2e-12, rtol=None):
    """
    Calculate an upper limit interval ``(0, poi_up)`` for a single
    Parameter of Interest (POI) using an automatic scan through
    POI-space, using the :func:`~scipy.optimize.toms748` algorithm.

    Example:
        >>> import numpy as np
        >>> import pyhf
        >>> pyhf.set_backend("numpy")
        >>> model = pyhf.simplemodels.hepdata_like(
        ...     signal_data=[12.0, 11.0], bkg_data=[50.0, 52.0], bkg_uncerts=[3.0, 7.0]
        ... )
        >>> observations = [51, 48]
        >>> data = pyhf.tensorlib.astensor(observations + model.config.auxdata)
        >>> obs_limit, exp_limits = pyhf.infer.intervals.upperlimit_auto(
        ...     data, model, 0., 5.
        ... )
        >>> obs_limit
        array(1.01156939)
        >>> exp_limits
        [array(0.55988001), array(0.75702336), array(1.06234693), array(1.50116923), array(2.05078596)]

    Args:
        data (:obj:`tensor`): The observed data.
        model (~pyhf.pdf.Model): The statistical model adhering to the schema ``model.json``.
        low (:obj:`float`): Lower boundary of search region
        high (:obj:`float`): Higher boundary of search region
        level (:obj:`float`): The threshold value to evaluate the interpolated results at.
                              Defaults to 0.05.
        atol (:obj:`float`): Absolute tolerance. Defaults to 1e-12. The iteration will end when the
                             result is within absolute *or* relative tolerance of the true limit.
        rtol (:obj:`float`): Relative tolerance. For optimal performance this argument should be set
                             to the highest acceptable relative tolerance, though it will default
                             to 1e-15 if not set.

    Returns:
        Tuple of Tensors:

            - Tensor: The observed upper limit on the POI.
            - Tensor: The expected upper limits on the POI.
    """
    if rtol is None:
        _warn(
            "upperlimit_auto: rtol not provided, defaulting to 1e-15. "
            "For optimal performance rtol should be set to the highest acceptable relative tolerance."
        )
        rtol = 1e-15

    cache = {}

    def f_all(mu):
        if mu in cache:
            return cache[mu]
        cache[mu] = hypotest(
            mu, data, model, test_stat="qtilde", return_expected_set=True
        )
        return cache[mu]

    def f(mu, limit=0):
        # Use integers for limit so we don't need a string comparison
        if limit == 0:
            # Obs
            return f_all(mu)[0] - level
        # Exp
        # (These are in the order -2, -1, 0, 1, 2 sigma)
        return f_all(mu)[1][limit - 1] - level

    def best_bracket(limit):
        # return best bracket
        ks = np.array(list(cache.keys()))
        vals = np.array(
            [
                v[0] - level if limit == 0 else v[1][limit - 1] - level
                for v in cache.values()
            ]
        )
        pos = vals >= 0
        neg = vals < 0
        lower = ks[pos][np.argmin(vals[pos])]
        upper = ks[neg][np.argmax(vals[neg])]
        return (lower, upper)

    tb, _ = get_backend()
    obs = tb.astensor(_toms748(f, low, high, args=(0), k=2, xtol=atol, rtol=rtol))
    exp = [
        tb.astensor(_toms748(f, *best_bracket(i), args=(i), k=2, xtol=atol, rtol=rtol))
        for i in range(1, 6)
    ]
    return obs, exp


def upperlimit(data, model, scan, level=0.05, return_results=False):
    """
    Calculate an upper limit interval ``(0, poi_up)`` for a single
    Parameter of Interest (POI) using a fixed scan through POI-space.

    Example:
        >>> import numpy as np
        >>> import pyhf
        >>> pyhf.set_backend("numpy")
        >>> model = pyhf.simplemodels.hepdata_like(
        ...     signal_data=[12.0, 11.0], bkg_data=[50.0, 52.0], bkg_uncerts=[3.0, 7.0]
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
        hypotest(mu, data, model, test_stat="qtilde", return_expected_set=True)
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
