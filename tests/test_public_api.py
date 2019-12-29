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


@pytest.mark.parametrize("backend_name", ["numpy", "tensorflow", "pytorch", "PyTorch"])
def test_set_backend_by_string(backend_name):
    pyhf.set_backend(backend_name)
    assert isinstance(
        pyhf.tensorlib,
        getattr(pyhf.tensor, "{0:s}_backend".format(backend_name.lower())),
    )


@pytest.mark.parametrize("backend_name", [b"numpy", b"tensorflow", b"pytorch"])
def test_set_backend_by_bytestring(backend_name):
    pyhf.set_backend(backend_name)
    assert isinstance(
        pyhf.tensorlib,
        getattr(pyhf.tensor, "{0:s}_backend".format(backend_name.decode("utf-8"))),
    )


@pytest.mark.parametrize("backend_name", ["fail", b"fail"])
def test_supported_backends(backend_name):
    with pytest.raises(pyhf.exceptions.InvalidBackend):
        pyhf.set_backend(backend_name)


def test_custom_backend_name_supported():
    class custom_backend(object):
        def __init__(self, **kwargs):
            self.name = "pytorch"

    with pytest.raises(AttributeError):
        pyhf.set_backend(custom_backend())


def test_custom_backend_name_notsupported():
    class custom_backend(object):
        def __init__(self, **kwargs):
            self.name = "notsupported"

    backend = custom_backend()
    assert pyhf.tensorlib.name != backend.name
    pyhf.set_backend(backend)
    assert pyhf.tensorlib.name == backend.name


def test_logpprob(backend, model_setup):
    model, data, init_pars = model_setup
    model.logpdf(init_pars, data)


def test_hypotest(backend, model_setup):
    model, data, init_pars = model_setup
    mu = 1.0
    pyhf.infer.hypotest(
        mu,
        data,
        model,
        init_pars,
        model.config.suggested_bounds(),
        return_expected_set=True,
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
    model = pyhf.simplemodels.hepdata_like(
        source['bindata']['sig'],
        source['bindata']['bkg'],
        source['bindata']['bkgerr'],
        batch_size=2,
    )

    pars = [model.config.suggested_init()] * 2
    data = source['bindata']['data'] + model.config.auxdata

    model.pdf(pars, data)
    model.expected_data(pars)
