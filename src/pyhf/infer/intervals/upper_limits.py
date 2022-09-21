"""Interval estimation"""
import numpy as np
from scipy.optimize import toms748

from pyhf import get_backend
from pyhf.infer import hypotest

__all__ = ["upper_limit", "linear_grid_scan", "toms748_scan"]


def __dir__():
    return __all__


def _interp(x, xp, fp):
    tb, _ = get_backend()
    return tb.astensor(np.interp(x, xp.tolist(), fp.tolist()))


def toms748_scan(
    data,
    model,
    bounds_low,
    bounds_up,
    level=0.05,
    atol=2e-12,
    rtol=1e-4,
    from_upper_limit_fn=False,
    **hypotest_kwargs,
):
    """
    Calculate an upper limit interval ``(0, poi_up)`` for a single
    Parameter of Interest (POI) using an automatic scan through
    POI-space, using the :func:`~scipy.optimize.toms748` algorithm.

    Example:
        >>> import numpy as np
        >>> import pyhf
        >>> pyhf.set_backend("numpy")
        >>> model = pyhf.simplemodels.uncorrelated_background(
        ...     signal=[12.0, 11.0], bkg=[50.0, 52.0], bkg_uncertainty=[3.0, 7.0]
        ... )
        >>> observations = [51, 48]
        >>> data = pyhf.tensorlib.astensor(observations + model.config.auxdata)
        >>> obs_limit, exp_limits = pyhf.infer.intervals.upper_limits.toms748_scan(
        ...     data, model, 0., 5., rtol=0.01
        ... )
        >>> obs_limit
        array(1.01156939)
        >>> exp_limits
        [array(0.5600747), array(0.75702605), array(1.06234693), array(1.50116923), array(2.05078912)]

    Args:
        data (:obj:`tensor`): The observed data.
        model (~pyhf.pdf.Model): The statistical model adhering to the schema ``model.json``.
        bounds_low (:obj:`float`): Lower boundary of search interval.
        bounds_up (:obj:`float`): Upper boundary of search interval.
        level (:obj:`float`): The threshold value to evaluate the interpolated results at.
                              Defaults to ``0.05``.
        atol (:obj:`float`): Absolute tolerance.
                             The iteration will end when the result is within absolute
                             *or* relative tolerance of the true limit.
        rtol (:obj:`float`): Relative tolerance.
                             For optimal performance this argument should be set
                             to the highest acceptable relative tolerance.
        hypotest_kwargs (:obj:`string`): Kwargs for the calls to
         :class:`~pyhf.infer.hypotest` to configure the fits.

    Returns:
        Tuple of Tensors:

            - Tensor: The observed upper limit on the POI.
            - Tensor: The expected upper limits on the POI.

    .. versionadded:: 0.7.0
    """
    cache = {}

    def f_cached(poi):
        if poi not in cache:
            cache[poi] = hypotest(
                poi,
                data,
                model,
                return_expected_set=True,
                **hypotest_kwargs,
            )
        return cache[poi]

    def f(poi, level, limit=0):
        # Use integers for limit so we don't need a string comparison
        # limit == 0: Observed
        # else: expected
        return (
            f_cached(poi)[0] - level
            if limit == 0
            else f_cached(poi)[1][limit - 1] - level
        )

    def best_bracket(limit):
        # return best bracket
        ks = np.asarray(list(cache.keys()))
        vals = np.asarray(
            [
                value[0] - level if limit == 0 else value[1][limit - 1] - level
                for value in cache.values()
            ]
        )
        pos = vals >= 0
        neg = vals < 0
        lower = ks[pos][np.argmin(vals[pos])]
        upper = ks[neg][np.argmax(vals[neg])]
        return (lower, upper)

    # extend bounds_low and bounds_up if they don't bracket CLs level
    lower_results = f_cached(bounds_low)
    # {lower,upper}_results[0] is an array and {lower,upper}_results[1] is a
    # list of arrays so need to turn {lower,upper}_results[0] into list to
    # concatenate them
    while np.any(np.asarray([lower_results[0]] + lower_results[1]) < level):
        bounds_low /= 2
        lower_results = f_cached(bounds_low)
    upper_results = f_cached(bounds_up)
    while np.any(np.asarray([upper_results[0]] + upper_results[1]) > level):
        bounds_up *= 2
        upper_results = f_cached(bounds_up)

    tb, _ = get_backend()
    obs = tb.astensor(
        toms748(f, bounds_low, bounds_up, args=(level, 0), k=2, xtol=atol, rtol=rtol)
    )
    exp = [
        tb.astensor(
            toms748(f, *best_bracket(idx), args=(level, idx), k=2, xtol=atol, rtol=rtol)
        )
        for idx in range(1, 6)
    ]
    if from_upper_limit_fn:
        return obs, exp, (list(cache.keys()), list(cache.values()))
    return obs, exp


def linear_grid_scan(
    data, model, scan, level=0.05, return_results=False, **hypotest_kwargs
):
    """
    Calculate an upper limit interval ``(0, poi_up)`` for a single
    Parameter of Interest (POI) using a linear scan through POI-space.

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
        >>> obs_limit, exp_limits, (scan, results) = pyhf.infer.intervals.upper_limits.upper_limit(
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

    .. versionadded:: 0.7.0
    """
    tb, _ = get_backend()
    results = [
        hypotest(mu, data, model, return_expected_set=True, **hypotest_kwargs)
        for mu in scan
    ]
    obs = tb.astensor([[r[0]] for r in results])
    exp = tb.astensor([[r[1][idx] for idx in range(5)] for r in results])

    result_array = tb.concatenate([obs, exp], axis=1).T

    # observed limit and the (0, +-1, +-2)sigma expected limits
    limits = [_interp(level, result_array[idx][::-1], scan[::-1]) for idx in range(6)]
    obs_limit, exp_limits = limits[0], limits[1:]

    if return_results:
        return obs_limit, exp_limits, (scan, results)
    return obs_limit, exp_limits


def upper_limit(
    data, model, scan=None, level=0.05, return_results=False, **hypotest_kwargs
):
    """
    Calculate an upper limit interval ``(0, poi_up)`` for a single Parameter of
    Interest (POI) using root-finding or a linear scan through POI-space.

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
        >>> obs_limit, exp_limits, (scan, results) = pyhf.infer.intervals.upper_limits.upper_limit(
        ...     data, model, scan, return_results=True
        ... )
        >>> obs_limit
        array(1.01764089)
        >>> exp_limits
        [array(0.59576921), array(0.76169166), array(1.08504773), array(1.50170482), array(2.06654952)]

    Args:
        data (:obj:`tensor`): The observed data.
        model (~pyhf.pdf.Model): The statistical model adhering to the schema ``model.json``.
        scan (:obj:`iterable` or ``None``): Iterable of POI values or ``None`` to use
         :class:`~pyhf.infer.intervals.upper_limits.toms748_scan`.
        level (:obj:`float`): The threshold value to evaluate the interpolated results at.
        return_results (:obj:`bool`): Whether to return the per-point results.

    Returns:
        Tuple of Tensors:

            - Tensor: The observed upper limit on the POI.
            - Tensor: The expected upper limits on the POI.
            - Tuple of Tensors: The given ``scan`` along with the
              :class:`~pyhf.infer.hypotest` results at each test POI.
              Only returned when ``return_results`` is ``True``.

    .. versionadded:: 0.7.0
    """
    if scan is not None:
        return linear_grid_scan(
            data, model, scan, level, return_results, **hypotest_kwargs
        )
    # else:
    bounds = model.config.suggested_bounds()[
        model.config.par_slice(model.config.poi_name).start
    ]
    obs_limit, exp_limit, results = toms748_scan(
        data,
        model,
        bounds[0],
        bounds[1],
        from_upper_limit_fn=True,
        **hypotest_kwargs,
    )
    if return_results:
        return obs_limit, exp_limit, results
    return obs_limit, exp_limit
