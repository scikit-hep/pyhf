import pytest
import pyhf
import numpy as np


@pytest.fixture(scope='function')
def model_setup(backend):
    np.random.seed(0)
    n_bins = 100
    model = pyhf.simplemodels.hepdata_like([10] * n_bins, [50] * n_bins, [1] * n_bins)
    init_pars = model.config.suggested_init()
    observations = np.random.randint(50, 60, size=n_bins).tolist()
    data = observations + model.config.auxdata
    return model, data, init_pars


def test_logpprob(backend, model_setup):
    model, data, init_pars = model_setup
    model.logpdf(init_pars, data)


def test_hypotest(backend, model_setup):
    model, data, init_pars = model_setup
    mu = 1.0
    pyhf.utils.hypotest(
        mu,
        data,
        model,
        init_pars,
        model.config.suggested_bounds(),
        return_expected_set=True,
        return_test_statistics=True,
    )


def test_prob_models(backend):
    tb, _ = backend
    pyhf.probability.Poisson(tb.astensor([10.0])).log_prob(tb.astensor(2.0))
    pyhf.probability.Normal(tb.astensor([10.0]), tb.astensor([1])).log_prob(
        tb.astensor(2.0)
    )


def test_pdf_batched(backend):
    tb, _ = backend
    source = {
        "binning": [2, -0.5, 1.5],
        "bindata": {"data": [55.0], "bkg": [50.0], "bkgerr": [7.0], "sig": [10.0]},
    }
    pdf = pyhf.simplemodels.hepdata_like(
        source['bindata']['sig'],
        source['bindata']['bkg'],
        source['bindata']['bkgerr'],
        batch_size=2,
    )

    pars = [pdf.config.suggested_init()] * 2
    data = source['bindata']['data'] + pdf.config.auxdata

    assert tb.tolist(pdf.pdf(pars, data)) == pytest.approx(
        [0.002417118312751542] * 2, 2.5e-05
    )
    assert tb.tolist(pdf.expected_data(pars))
    assert tb.tolist(pdf.expected_data(pars)[0]) == pytest.approx(
        [60.0, 51.020408630], 1e-08
    )
    assert tb.tolist(pdf.expected_data(pars)[1]) == pytest.approx(
        [60.0, 51.020408630], 1e-08
    )
