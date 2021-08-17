import pytest

import pyhf
import pyhf.infer.calculators


def test_calc_dist():
    asymptotic_dist = pyhf.infer.calculators.AsymptoticTestStatDistribution(0.0)
    assert asymptotic_dist.pvalue(-1) == 1 - asymptotic_dist.cdf(-1)


@pytest.mark.parametrize("return_fitted_pars", [False, True])
def test_generate_asimov_can_return_fitted_pars(return_fitted_pars):
    model = pyhf.simplemodels.uncorrelated_background([1, 1], [1, 1], [1, 1])
    data = [2, 2, 1, 1]  # [main x 2, aux x 2]
    init_pars = model.config.suggested_init()
    par_bounds = model.config.suggested_bounds()
    fixed_params = model.config.suggested_fixed()

    result = pyhf.infer.calculators.generate_asimov_data(
        1.0,
        data,
        model,
        init_pars,
        par_bounds,
        fixed_params,
        return_fitted_pars=return_fitted_pars,
    )

    if return_fitted_pars:
        assert len(result) == 2
        result, asimov_pars = result
        assert pytest.approx([1.0, 1.0, 1.0]) == pyhf.tensorlib.tolist(asimov_pars)
    assert pytest.approx([2.0, 2.0, 1.0, 1.0]) == pyhf.tensorlib.tolist(result)
