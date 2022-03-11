"""Significance calculations under Normal hypothesis"""
from pyhf.infer import hypotest
from scipy.stats import norm

__all__ = ["discovery_significance"]


def __dir__():
    return __all__


from pyhf.infer import _check_hypotest_prerequisites
from pyhf.infer import utils


def _minimum_hypotest_api(
    data,
    model,
    init_pars=None,
    par_bounds=None,
    fixed_params=None,
    calctype="asymptotics",
    **kwargs,
):
    init_pars = init_pars or model.config.suggested_init()
    par_bounds = par_bounds or model.config.suggested_bounds()
    fixed_params = fixed_params or model.config.suggested_fixed()

    _check_hypotest_prerequisites(model, data, init_pars, par_bounds, fixed_params)

    return utils.create_calculator(
        calctype,
        data,
        model,
        init_pars,
        par_bounds,
        fixed_params,
        **kwargs,
    )


def _minimum_discovery_significance(
    data, model, return_expected=True, return_observed=False, **hypotest_kwargs
):

    init_pars = hypotest_kwargs.pop("init_pars", None)
    par_bounds = hypotest_kwargs.pop("par_bounds", None)
    fixed_params = hypotest_kwargs.pop("fixed_params", None)
    calctype = hypotest_kwargs.pop("calctype", "asymptotics")

    calculator = _minimum_hypotest_api(
        data,
        model,
        init_pars=init_pars,
        par_bounds=par_bounds,
        fixed_params=fixed_params,
        calctype=calctype,
        test_stat="q0",
    )

    test_poi = 0.0  # discovery
    test_stat = calculator.teststatistic(test_poi)
    sig_plus_bkg_distribution, bkg_only_distribution = calculator.distributions(
        test_poi
    )

    # Expected
    if return_expected:
        cl_sb_exp_band, _, _ = calculator.expected_pvalues(
            sig_plus_bkg_distribution, bkg_only_distribution
        )
        cl_sb_exp = cl_sb_exp_band[2]  # Median expected p-value
        expected_significance = norm.isf(float(cl_sb_exp), loc=0, scale=1)

        print(f"Expected significance: {expected_significance}")

    # Observed
    if return_observed:
        cl_sb_obs, _, _ = calculator.pvalues(
            test_stat, sig_plus_bkg_distribution, bkg_only_distribution
        )
        observed_significance = norm.isf(float(cl_sb_obs), loc=0, scale=1)
        print(f"Observed significance: {observed_significance}")


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
