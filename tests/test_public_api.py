import pytest
import pyhf
import numpy as np


@pytest.fixture(scope='function')
def model_setup(backend):
    np.random.seed(0)
    n_bins = 100
    # TODO: Simplify after pyhf v0.6.2 released
    if tuple(int(part) for part in pyhf.__version__.split(".")[:3]) < (0, 6, 2):
        model = pyhf.simplemodels.hepdata_like(
            [10] * n_bins, [50] * n_bins, [1] * n_bins
        )
    else:
        model = pyhf.simplemodels.uncorrelated_background(
            [10] * n_bins, [50] * n_bins, [1] * n_bins
        )
    init_pars = model.config.suggested_init()
    observations = np.random.randint(50, 60, size=n_bins).tolist()
    data = observations + model.config.auxdata
    return model, data, init_pars


@pytest.mark.parametrize("backend_name", ["numpy", "tensorflow", "pytorch", "PyTorch"])
def test_set_backend_by_string(backend_name):
    pyhf.set_backend(backend_name)
    assert isinstance(
        pyhf.tensorlib,
        getattr(pyhf.tensor, f"{backend_name.lower():s}_backend"),
    )


@pytest.mark.parametrize("optimizer_name", ["scipy", "minuit"])
def test_set_optimizer_by_string(optimizer_name):
    pyhf.set_backend(pyhf.tensorlib, optimizer_name)
    assert isinstance(
        pyhf.optimizer,
        getattr(pyhf.optimize, f"{optimizer_name.lower():s}_optimizer"),
    )


@pytest.mark.parametrize("precision_level", ["32b", "64b"])
def test_set_precision_by_string(precision_level):
    pyhf.set_backend(pyhf.tensorlib.name, precision=precision_level)
    assert pyhf.tensorlib.precision == precision_level.lower()
    pyhf.set_backend(pyhf.tensor.numpy_backend(precision=precision_level))
    assert pyhf.tensorlib.precision == precision_level.lower()


@pytest.mark.parametrize("backend_name", [b"numpy", b"tensorflow", b"pytorch"])
def test_set_backend_by_bytestring(backend_name):
    pyhf.set_backend(backend_name)
    assert isinstance(
        pyhf.tensorlib,
        getattr(pyhf.tensor, f"{backend_name.decode('utf-8'):s}_backend"),
    )


@pytest.mark.parametrize("optimizer_name", [b"scipy", b"minuit"])
def test_set_optimizer_by_bytestring(optimizer_name):
    pyhf.set_backend(pyhf.tensorlib, optimizer_name)
    assert isinstance(
        pyhf.optimizer,
        getattr(pyhf.optimize, f"{optimizer_name.decode('utf-8'):s}_optimizer"),
    )


@pytest.mark.parametrize("precision_level", [b"32b", b"64b"])
def test_set_precision_by_bytestring(precision_level):
    pyhf.set_backend(pyhf.tensorlib.name, precision=precision_level)
    assert pyhf.tensorlib.precision == precision_level.decode("utf-8")


@pytest.mark.parametrize("precision_level", ["32b", "64b"])
def test_set_precision_by_string_wins(precision_level):
    conflicting_precision = "32b" if precision_level == "64b" else "64b"
    pyhf.set_backend(
        pyhf.tensor.numpy_backend(precision=conflicting_precision),
        precision=precision_level,
    )
    assert pyhf.tensorlib.precision == precision_level.lower()


@pytest.mark.parametrize("backend_name", ["fail", b"fail"])
def test_supported_backends(backend_name):
    with pytest.raises(pyhf.exceptions.InvalidBackend):
        pyhf.set_backend(backend_name)


@pytest.mark.parametrize("optimizer_name", ["fail", b"fail"])
def test_supported_optimizers(optimizer_name):
    with pytest.raises(pyhf.exceptions.InvalidOptimizer):
        pyhf.set_backend(pyhf.tensorlib, optimizer_name)


@pytest.mark.parametrize("precision_level", ["fail", b"fail"])
def test_supported_precision(precision_level):
    with pytest.raises(pyhf.exceptions.Unsupported):
        pyhf.set_backend("numpy", precision=precision_level)


def test_custom_backend_name_supported():
    class custom_backend:
        def __init__(self, **kwargs):
            self.name = "pytorch"
            self.precision = '64b'

        def _setup(self):
            pass

    with pytest.raises(AttributeError):
        pyhf.set_backend(custom_backend())


def test_custom_optimizer_name_supported():
    class custom_optimizer:
        def __init__(self, **kwargs):
            self.name = "scipy"

    with pytest.raises(AttributeError):
        pyhf.set_backend(pyhf.tensorlib, custom_optimizer())


def test_custom_backend_name_notsupported():
    class custom_backend:
        def __init__(self, **kwargs):
            self.name = "notsupported"
            self.precision = '64b'

        def _setup(self):
            pass

    backend = custom_backend()
    assert pyhf.tensorlib.name != backend.name
    pyhf.set_backend(backend)
    assert pyhf.tensorlib.name == backend.name


def test_custom_optimizer_name_notsupported():
    class custom_optimizer:
        def __init__(self, **kwargs):
            self.name = "notsupported"

    optimizer = custom_optimizer()
    assert pyhf.optimizer.name != optimizer.name
    pyhf.set_backend(pyhf.tensorlib, optimizer)
    assert pyhf.optimizer.name == optimizer.name


@pytest.mark.parametrize("backend_name", ["numpy", "tensorflow", "pytorch", "PyTorch"])
def test_backend_no_custom_attributes(backend_name):
    pyhf.set_backend(backend_name)
    with pytest.raises(AttributeError):
        pyhf.tensorlib.nonslotted = True


@pytest.mark.parametrize("backend_name", ["numpy", "tensorflow", "pytorch", "PyTorch"])
def test_backend_slotted_attributes(backend_name):
    pyhf.set_backend(backend_name)
    for attr in ["name", "precision", "dtypemap", "default_do_grad"]:
        assert getattr(pyhf.tensorlib, attr) is not None


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


# TODO: Remove after pyhf v0.6.2 released
def test_pdf_batched_deprecated_api(backend):
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


# TODO: Remove skipif after pyhf v0.6.2 released
@pytest.mark.skipif(
    tuple(int(part) for part in pyhf.__version__.split(".")[:3]) < (0, 6, 2),
    reason="requires pyhf v0.6.2+",
)
def test_pdf_batched(backend):
    tb, _ = backend
    source = {
        "binning": [2, -0.5, 1.5],
        "bindata": {"data": [55.0], "bkg": [50.0], "bkgerr": [7.0], "sig": [10.0]},
    }
    model = pyhf.simplemodels.uncorrelated_background(
        source['bindata']['sig'],
        source['bindata']['bkg'],
        source['bindata']['bkgerr'],
        batch_size=2,
    )

    pars = [model.config.suggested_init()] * 2
    data = source['bindata']['data'] + model.config.auxdata

    model.pdf(pars, data)
    model.expected_data(pars)
