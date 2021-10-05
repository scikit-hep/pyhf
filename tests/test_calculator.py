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


# test different test stats because those affect the control flow
# in AsymptotiCalculator.teststatistic, where the fit results should be set
# the other kwargs don't impact the logic of that method,
# so leave them at the default so as not to put a burden on future changes
@pytest.mark.parametrize('test_stat', ['qtilde', 'q', 'q0'])
def test_asymptotic_calculator_has_fitted_pars(test_stat):
    model = pyhf.simplemodels.uncorrelated_background([1], [1], [1])
    data = [2, 1]  # [main, aux]

    calc = pyhf.infer.calculators.AsymptoticCalculator(data, model, test_stat=test_stat)
    calc.teststatistic(0 if test_stat == 'q0' else 1)

    assert hasattr(calc, 'fitted_pars')
    fitted_pars = calc.fitted_pars
    assert hasattr(fitted_pars, 'asimov_pars')
    assert hasattr(fitted_pars, 'fixed_poi_fit_to_data')
    assert hasattr(fitted_pars, 'fixed_poi_fit_to_asimov')
    assert hasattr(fitted_pars, 'free_fit_to_data')
    assert hasattr(fitted_pars, 'free_fit_to_asimov')

    rtol = 1e-5
    if test_stat == 'q0':
        assert pytest.approx([1.0, 1.0], rel=rtol) == pyhf.tensorlib.tolist(
            fitted_pars.asimov_pars
        )
        assert pytest.approx([0.0, 1.5], rel=rtol) == pyhf.tensorlib.tolist(
            fitted_pars.fixed_poi_fit_to_data
        )
        assert pytest.approx([0.0, 1.5], rel=rtol) == pyhf.tensorlib.tolist(
            fitted_pars.fixed_poi_fit_to_asimov
        )
        assert pytest.approx([1.0, 1.0], rel=rtol) == pyhf.tensorlib.tolist(
            fitted_pars.free_fit_to_data
        )
        assert pytest.approx([1.0, 1.0], rel=rtol) == pyhf.tensorlib.tolist(
            fitted_pars.free_fit_to_asimov
        )
    else:
        assert pytest.approx([0.0, 1.5], rel=rtol) == pyhf.tensorlib.tolist(
            fitted_pars.asimov_pars
        )
        assert pytest.approx([1.0, 1.0], rel=rtol) == pyhf.tensorlib.tolist(
            fitted_pars.fixed_poi_fit_to_data
        )
        assert pytest.approx([1.0, 1.1513553], rel=rtol) == pyhf.tensorlib.tolist(
            fitted_pars.fixed_poi_fit_to_asimov
        )
        assert pytest.approx([1.0, 1.0], rel=rtol) == pyhf.tensorlib.tolist(
            fitted_pars.free_fit_to_data
        )
        assert pytest.approx(
            [7.6470499e-05, 1.4997178], rel=rtol
        ) == pyhf.tensorlib.tolist(fitted_pars.free_fit_to_asimov)
