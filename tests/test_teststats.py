import logging
import math

import pytest

import pyhf
import pyhf.infer.test_statistics


def test_q0(caplog):
    mu = 1.0
    model = pyhf.simplemodels.uncorrelated_background([6], [9], [3])
    data = [9] + model.config.auxdata
    init_pars = model.config.suggested_init()
    par_bounds = model.config.suggested_bounds()
    fixed_params = model.config.suggested_fixed()

    with caplog.at_level(logging.WARNING, "pyhf.infer.test_statistics"):
        pyhf.infer.test_statistics.q0(
            mu, data, model, init_pars, par_bounds, fixed_params
        )
        assert (
            "q0 test statistic only used for fit configuration with POI set to zero"
            in caplog.text
        )
        caplog.clear()


def test_qmu(caplog):
    mu = 1.0
    model = pyhf.simplemodels.uncorrelated_background([6], [9], [3])
    data = [9] + model.config.auxdata
    init_pars = model.config.suggested_init()
    par_bounds = model.config.suggested_bounds()
    fixed_params = model.config.suggested_fixed()

    with caplog.at_level(logging.WARNING, "pyhf.infer.test_statistics"):
        pyhf.infer.test_statistics.qmu(
            mu, data, model, init_pars, par_bounds, fixed_params
        )
        assert "qmu test statistic used for fit" in caplog.text
        caplog.clear()


def test_qmu_tilde(caplog):
    mu = 1.0
    model = pyhf.simplemodels.uncorrelated_background([6], [9], [3])
    data = [9] + model.config.auxdata
    init_pars = model.config.suggested_init()
    par_bounds = model.config.suggested_bounds()
    fixed_params = model.config.suggested_fixed()

    par_bounds[model.config.poi_index] = [-10, 10]
    with caplog.at_level(logging.WARNING, "pyhf.infer.test_statistics"):
        pyhf.infer.test_statistics.qmu_tilde(
            mu, data, model, init_pars, par_bounds, fixed_params
        )
        assert "qmu_tilde test statistic used for fit" in caplog.text
        caplog.clear()


def test_tmu(caplog):
    mu = 1.0
    model = pyhf.simplemodels.uncorrelated_background([6], [9], [3])
    data = [9] + model.config.auxdata
    init_pars = model.config.suggested_init()
    par_bounds = model.config.suggested_bounds()
    fixed_params = model.config.suggested_fixed()

    with caplog.at_level(logging.WARNING, "pyhf.infer.test_statistics"):
        pyhf.infer.test_statistics.tmu(
            mu, data, model, init_pars, par_bounds, fixed_params
        )
        assert "tmu test statistic used for fit" in caplog.text
        caplog.clear()


def test_tmu_tilde(caplog):
    mu = 1.0
    model = pyhf.simplemodels.uncorrelated_background([6], [9], [3])
    data = [9] + model.config.auxdata
    init_pars = model.config.suggested_init()
    par_bounds = model.config.suggested_bounds()
    fixed_params = model.config.suggested_fixed()

    par_bounds[model.config.poi_index] = [-10, 10]
    with caplog.at_level(logging.WARNING, "pyhf.infer.test_statistics"):
        pyhf.infer.test_statistics.tmu_tilde(
            mu, data, model, init_pars, par_bounds, fixed_params
        )
        assert "tmu_tilde test statistic used for fit" in caplog.text
        caplog.clear()


@pytest.mark.parametrize(
    "test_stat",
    [pyhf.infer.test_statistics.qmu, pyhf.infer.test_statistics.tmu],
    ids=["qmu", "tmu"],
)
def test_unbounded_poi_test_stats(caplog, test_stat):
    """
    Check qmu and tmu for a POI with a negative lower bound (no warning path)
    against the closed-form profile likelihood ratio.
    """
    mu = 1.0
    model = pyhf.simplemodels.uncorrelated_background([6], [9], [3])
    data = [9] + model.config.auxdata
    init_pars = model.config.suggested_init()
    par_bounds = model.config.suggested_bounds()
    par_bounds[model.config.poi_index] = [-10.0, 10.0]
    fixed_params = model.config.suggested_fixed()

    with caplog.at_level(logging.WARNING, "pyhf.infer.test_statistics"):
        result = test_stat(mu, data, model, init_pars, par_bounds, fixed_params)
        assert "test statistic used for fit" not in caplog.text
        caplog.clear()

    # Observed n = 9 with background b = 9 and auxiliary data tau = (9 / 3)^2 = 9
    # give the unconstrained best fit mu_hat = 0, gamma_hat = 1.
    # For mu = 1 (signal s = 6) the profiled gamma solves 3 gamma^2 - gamma - 1 = 0.
    def log_likelihood(mu, gamma):
        main_rate = 6.0 * mu + 9.0 * gamma
        aux_rate = 9.0 * gamma
        return (
            9.0 * math.log(main_rate) - main_rate + 9.0 * math.log(aux_rate) - aux_rate
        )

    gamma_hat_hat = (1.0 + math.sqrt(13.0)) / 6.0
    # mu_hat = 0 <= mu so qmu equals tmu
    expected = 2.0 * (log_likelihood(0.0, 1.0) - log_likelihood(mu, gamma_hat_hat))
    assert pyhf.tensorlib.tolist(result) == pytest.approx(expected, rel=1e-4)


def test_no_poi_test_stats():
    spec = {
        "channels": [
            {
                "name": "channel",
                "samples": [
                    {
                        "name": "sample",
                        "data": [10.0],
                        "modifiers": [
                            {
                                "type": "normsys",
                                "name": "shape",
                                "data": {"hi": 0.5, "lo": 1.5},
                            }
                        ],
                    },
                ],
            }
        ]
    }
    model = pyhf.Model(spec, poi_name=None)

    test_poi = 1.0
    data = [12] + model.config.auxdata
    init_pars = model.config.suggested_init()
    par_bounds = model.config.suggested_bounds()
    fixed_params = model.config.suggested_fixed()

    with pytest.raises(
        pyhf.exceptions.UnspecifiedPOI,
        match=r"No POI is defined. A POI is required for profile likelihood based test statistics.",
    ):
        pyhf.infer.test_statistics.q0(
            test_poi, data, model, init_pars, par_bounds, fixed_params
        )

    with pytest.raises(
        pyhf.exceptions.UnspecifiedPOI,
        match=r"No POI is defined. A POI is required for profile likelihood based test statistics.",
    ):
        pyhf.infer.test_statistics.qmu(
            test_poi, data, model, init_pars, par_bounds, fixed_params
        )

    with pytest.raises(
        pyhf.exceptions.UnspecifiedPOI,
        match=r"No POI is defined. A POI is required for profile likelihood based test statistics.",
    ):
        pyhf.infer.test_statistics.qmu_tilde(
            test_poi, data, model, init_pars, par_bounds, fixed_params
        )

    with pytest.raises(
        pyhf.exceptions.UnspecifiedPOI,
        match=r"No POI is defined. A POI is required for profile likelihood based test statistics.",
    ):
        pyhf.infer.test_statistics.tmu(
            test_poi, data, model, init_pars, par_bounds, fixed_params
        )

    with pytest.raises(
        pyhf.exceptions.UnspecifiedPOI,
        match=r"No POI is defined. A POI is required for profile likelihood based test statistics.",
    ):
        pyhf.infer.test_statistics.tmu_tilde(
            test_poi, data, model, init_pars, par_bounds, fixed_params
        )


@pytest.mark.parametrize("test_stat", ["qtilde", "q"])
def test_get_teststat_by_name(test_stat):
    assert pyhf.infer.utils.get_test_stat(test_stat)


def test_get_teststat_error():
    with pytest.raises(pyhf.exceptions.InvalidTestStatistic):
        pyhf.infer.utils.get_test_stat("look at me i'm not real")


@pytest.mark.parametrize("return_fitted_pars", [False, True])
@pytest.mark.parametrize(
    "test_stat",
    [
        pyhf.infer.test_statistics.q0,
        pyhf.infer.test_statistics.qmu,
        pyhf.infer.test_statistics.qmu_tilde,
        pyhf.infer.test_statistics.tmu,
        pyhf.infer.test_statistics.tmu_tilde,
    ],
)
def test_return_fitted_pars(test_stat, return_fitted_pars):
    mu = 0.0 if test_stat is pyhf.infer.test_statistics.q0 else 1.0
    model = pyhf.simplemodels.uncorrelated_background([6], [9], [3])
    data = [9] + model.config.auxdata
    init_pars = model.config.suggested_init()
    par_bounds = model.config.suggested_bounds()
    fixed_params = model.config.suggested_fixed()

    result = test_stat(
        mu,
        data,
        model,
        init_pars,
        par_bounds,
        fixed_params,
        return_fitted_pars=return_fitted_pars,
    )
    if return_fitted_pars:
        assert len(result) == 2
        assert len(result[1]) == 2
        result, (pars_bestfit, pars_constrained_fit) = result
        assert len(pars_bestfit) == len(init_pars)
        assert len(pars_constrained_fit) == len(init_pars)
    assert result > -1e4  # >= 0 but with generous tolerance
