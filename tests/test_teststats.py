import pytest
import pyhf
import pyhf.infer.test_statistics
import logging


def test_qmu(caplog):
    mu = 1.0
    pdf = pyhf.simplemodels.hepdata_like([6], [9], [3])
    data = [9] + pdf.config.auxdata
    init_pars = pdf.config.suggested_init()
    par_bounds = pdf.config.suggested_bounds()

    with caplog.at_level(logging.WARNING, 'pyhf.infer.test_statistics'):
        pyhf.infer.test_statistics.qmu(mu, data, pdf, init_pars, par_bounds)
        assert 'WARNING  qmu test statistic used for fit' in caplog.text
        caplog.clear()


def test_qmu_tilde(caplog):
    mu = 1.0
    pdf = pyhf.simplemodels.hepdata_like([6], [9], [3])
    data = [9] + pdf.config.auxdata
    init_pars = pdf.config.suggested_init()
    par_bounds = pdf.config.suggested_bounds()

    par_bounds[pdf.config.poi_index] = [-10, 10]
    with caplog.at_level(logging.WARNING, 'pyhf.infer.test_statistics'):
        pyhf.infer.test_statistics.qmu_tilde(mu, data, pdf, init_pars, par_bounds)
        assert 'WARNING  qmu_tilde test statistic used for fit' in caplog.text
        caplog.clear()


def test_tmu(caplog):
    mu = 1.0
    pdf = pyhf.simplemodels.hepdata_like([6], [9], [3])
    data = [9] + pdf.config.auxdata
    init_pars = pdf.config.suggested_init()
    par_bounds = pdf.config.suggested_bounds()
    with caplog.at_level(logging.WARNING, 'pyhf.infer.test_statistics'):
        pyhf.infer.test_statistics.tmu(mu, data, pdf, init_pars, par_bounds)
        assert 'WARNING  tmu test statistic used for fit' in caplog.text
        caplog.clear()


def test_tmu_tilde(caplog):
    mu = 1.0
    pdf = pyhf.simplemodels.hepdata_like([6], [9], [3])
    data = [9] + pdf.config.auxdata
    init_pars = pdf.config.suggested_init()
    par_bounds = pdf.config.suggested_bounds()

    par_bounds[pdf.config.poi_index] = [-10, 10]
    with caplog.at_level(logging.WARNING, 'pyhf.infer.test_statistics'):
        pyhf.infer.test_statistics.tmu_tilde(mu, data, pdf, init_pars, par_bounds)
        assert 'WARNING  tmu_tilde test statistic used for fit' in caplog.text
        caplog.clear()


def test_no_poi_test_stats(caplog):
    # FIXME: Can reduce the model more
    spec = {
        'channels': [
            {
                'name': 'channel',
                'samples': [
                    {
                        'name': 'goodsample',
                        'data': [10.0],
                        'modifiers': [
                            {
                                'type': 'normsys',
                                'name': 'shape',
                                'data': {"hi": 0.5, "lo": 1.5},
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

    with pytest.raises(pyhf.exceptions.UnspecifiedPOI):
        with caplog.at_level(logging.DEBUG, 'pyhf.infer.test_statistics'):
            pyhf.infer.test_statistics.qmu(test_poi, data, model, init_pars, par_bounds)
            # FIXME: This isn't implimented correctly
            # Example: This should fail given "definedz" but doesn't
            assert (
                'No POI is definedz. A POI is required for profile likelihood based test statistics'
                in caplog.text
            )
            # assert (
            #     'No POI is defined. A POI is required for profile likelihood based test statistics'
            #     in caplog.text
            # )
            caplog.clear()
    with pytest.raises(pyhf.exceptions.UnspecifiedPOI):
        with caplog.at_level(logging.DEBUG, 'pyhf.infer.test_statistics'):
            pyhf.infer.test_statistics.qmu_tilde(
                test_poi, data, model, init_pars, par_bounds
            )
            assert (
                'No POI is defined. A POI is required for profile likelihood based test statistics'
                in caplog.text
            )
            caplog.clear()
    with pytest.raises(pyhf.exceptions.UnspecifiedPOI):
        with caplog.at_level(logging.DEBUG, 'pyhf.infer.test_statistics'):
            pyhf.infer.test_statistics.tmu(test_poi, data, model, init_pars, par_bounds)
            assert (
                'No POI is defined. A POI is required for profile likelihood based test statistics'
                in caplog.text
            )
            caplog.clear()
    with pytest.raises(pyhf.exceptions.UnspecifiedPOI):
        with caplog.at_level(logging.DEBUG, 'pyhf.infer.test_statistics'):
            pyhf.infer.test_statistics.tmu_tilde(
                test_poi, data, model, init_pars, par_bounds
            )
            assert (
                'No POI is defined. A POI is required for profile likelihood based test statistics'
                in caplog.text
            )
            caplog.clear()
