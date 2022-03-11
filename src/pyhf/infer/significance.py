"""Significance calculations under Normal hypothesis"""
from pyhf.infer import hypotest
from scipy.stats import norm

__all__ = ["discovery_significance"]


def __dir__():
    return __all__


def discovery_significance(data, model, return_expected=True, **hypotest_kwargs):
    r"""
    The discovery significance
    """

    init_pars = hypotest_kwargs.pop("init_pars", None)
    fixed_params = hypotest_kwargs.pop("fixed_params", None)
    par_bounds = hypotest_kwargs.pop("par_bounds", None)
    hypotest_kwargs.pop("return_expected", None)  # Force this to arg

    discovery_p_values = hypotest(
        0.0,  # discovery
        data,
        model,
        init_pars=init_pars,
        fixed_params=fixed_params,
        par_bounds=par_bounds,
        test_stat="q0",
        return_expected=return_expected,
        **hypotest_kwargs,
    )

    if return_expected:
        obs_p_value, exp_p_value = discovery_p_values
        expected_significance = norm.isf(float(exp_p_value), loc=0, scale=1)
    else:
        obs_p_value = discovery_p_values

    observed_significance = norm.isf(float(obs_p_value), loc=0, scale=1)

    return (
        (observed_significance, expected_significance)
        if return_expected
        else observed_significance
    )
