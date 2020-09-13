import pytest
import pyhf
import pyhf.infer.test_statistics
import logging


def test_qmu(caplog):
    mu = 1.0
    model = pyhf.simplemodels.hepdata_like([6], [9], [3])
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
    model = pyhf.simplemodels.hepdata_like([6], [9], [3])
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
    model = pyhf.simplemodels.hepdata_like([6], [9], [3])
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
    model = pyhf.simplemodels.hepdata_like([6], [9], [3])
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

    with pytest.raises(pyhf.exceptions.UnspecifiedPOI) as excinfo:
        pyhf.infer.test_statistics.qmu(
            test_poi, data, model, init_pars, par_bounds, fixed_params
        )
    assert (
        "No POI is defined. A POI is required for profile likelihood based test statistics."
        in str(excinfo.value)
    )

    with pytest.raises(pyhf.exceptions.UnspecifiedPOI) as excinfo:
        pyhf.infer.test_statistics.qmu_tilde(
            test_poi, data, model, init_pars, par_bounds, fixed_params
        )
    assert (
        "No POI is defined. A POI is required for profile likelihood based test statistics."
        in str(excinfo.value)
    )

    with pytest.raises(pyhf.exceptions.UnspecifiedPOI) as excinfo:
        pyhf.infer.test_statistics.tmu(
            test_poi, data, model, init_pars, par_bounds, fixed_params
        )
    assert (
        "No POI is defined. A POI is required for profile likelihood based test statistics."
        in str(excinfo.value)
    )

    with pytest.raises(pyhf.exceptions.UnspecifiedPOI) as excinfo:
        pyhf.infer.test_statistics.tmu_tilde(
            test_poi, data, model, init_pars, par_bounds, fixed_params
        )
    assert (
        "No POI is defined. A POI is required for profile likelihood based test statistics."
        in str(excinfo.value)
    )
