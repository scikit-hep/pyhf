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


def test_qmu_tilde(caplog):
    mu = 1.0
    pdf = pyhf.simplemodels.hepdata_like([6], [9], [3])
    data = [9] + pdf.config.auxdata
    init_pars = pdf.config.suggested_init()
    par_bounds = pdf.config.suggested_bounds()

    par_bounds[pdf.config.poi_index] = [-10, 10]
    with caplog.at_level(logging.WARNING, 'pyhf.infer.test_statistics'):
        pyhf.infer.test_statistics.qmu_tilde(mu, data, pdf, init_pars, par_bounds)
        assert 'WARNING  qmu tilde test statistic used for fit' in caplog.text


def test_tmu(caplog):
    mu = 1.0
    pdf = pyhf.simplemodels.hepdata_like([6], [9], [3])
    data = [9] + pdf.config.auxdata
    init_pars = pdf.config.suggested_init()
    par_bounds = pdf.config.suggested_bounds()
    with caplog.at_level(logging.WARNING, 'pyhf.infer.test_statistics'):
        pyhf.infer.test_statistics.tmu(mu, data, pdf, init_pars, par_bounds)
        assert 'WARNING  tmu test statistic used for fit' in caplog.text

    par_bounds[pdf.config.poi_index] = [-10, 10]


def test_tmu_tilde(caplog):
    mu = 1.0
    pdf = pyhf.simplemodels.hepdata_like([6], [9], [3])
    data = [9] + pdf.config.auxdata
    init_pars = pdf.config.suggested_init()
    par_bounds = pdf.config.suggested_bounds()

    par_bounds[pdf.config.poi_index] = [-10, 10]
    with caplog.at_level(logging.WARNING, 'pyhf.infer.test_statistics'):
        pyhf.infer.test_statistics.tmu_tilde(mu, data, pdf, init_pars, par_bounds)
        assert 'WARNING  tmu tilde test statistic used for fit' in caplog.text
