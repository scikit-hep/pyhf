"""Interval estimation"""
import pyhf.infer.intervals.upper_limits

__all__ = ["upper_limits.upper_limit"]


def __dir__():
    return __all__


def upperlimit(
    data, model, scan=None, level=0.05, return_results=False, **hypotest_kwargs
):
    """
    .. deprecated:: 0.7.0
       Use :func:`~pyhf.infer.intervals.upper_limits.upper_limit` instead.
    .. warning:: :func:`~pyhf.infer.intervals.upperlimit` will be removed in
     ``pyhf`` ``v0.9.0``.
    """
    from pyhf.exceptions import _deprecated_api_warning

    _deprecated_api_warning(
        "pyhf.infer.intervals.upperlimit",
        "pyhf.infer.intervals.upper_limits.upper_limit",
        "0.7.0",
        "0.9.0",
    )
    return pyhf.infer.intervals.upper_limits.upper_limit(
        data, model, scan, level, return_results, **hypotest_kwargs
    )
