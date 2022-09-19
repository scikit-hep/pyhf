"""Interval estimation"""
import pyhf.infer.intervals.upper_limits

__all__ = ["upper_limits.upper_limit"]


def __dir__():
    return __all__


def upperlimit(
    data, model, scan=None, level=0.05, return_results=False, **hypotest_kwargs
):
    # Warn here
    return pyhf.infer.intervals.upper_limits.upper_limit(
        data, model, scan, level, return_results, **hypotest_kwargs
    )
