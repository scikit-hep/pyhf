import pytest
import sys
import pyhf


@pytest.mark.parametrize(
    "param",
    [
        ["numpy", "numpy_backend", "numpy_backend", pytest.raises(ImportError)],
        [
            "torch",
            "pytorch_backend",
            "pytorch_backend",
            pytest.raises(pyhf.exceptions.ImportBackendError),
        ],
        [
            "tensorflow",
            "tensorflow_backend",
            "tensorflow_backend",
            pytest.raises(pyhf.exceptions.ImportBackendError),
        ],
        [
            "jax",
            "jax_backend",
            "jax_backend",
            pytest.raises(pyhf.exceptions.ImportBackendError),
        ],
    ],
    ids=["numpy", "pytorch", "tensorflow", "jax"],
)
def test_missing_backends(isolate_modules, param):
    backend_name, module_name, import_name, expectation = param

    # hide
    CACHE_BACKEND, sys.modules[backend_name] = sys.modules[backend_name], None
    sys.modules.setdefault(f'pyhf.tensor.{import_name}', None)
    CACHE_MODULE, sys.modules[f'pyhf.tensor.{module_name}'] = (
        sys.modules[f'pyhf.tensor.{module_name}'],
        None,
    )

    try:
        delattr(pyhf.tensor, module_name)
    except:  # noqa: E722
        pass

    with expectation:
        getattr(pyhf.tensor, module_name)

    # put back
    CACHE_BACKEND, sys.modules[backend_name] = None, CACHE_BACKEND
    CACHE_MODULE, sys.modules[f'pyhf.tensor.{module_name}'] = (
        None,
        CACHE_MODULE,
    )


@pytest.mark.parametrize(
    "param",
    [
        ["scipy", "scipy_optimizer", "opt_scipy", pytest.raises(ImportError)],
        [
            "iminuit",
            "minuit_optimizer",
            "opt_minuit",
            pytest.raises(pyhf.exceptions.ImportBackendError),
        ],
    ],
    ids=["scipy", "minuit"],
)
def test_missing_optimizer(isolate_modules, param):
    backend_name, module_name, import_name, expectation = param

    # hide
    CACHE_BACKEND, sys.modules[backend_name] = sys.modules[backend_name], None
    sys.modules.setdefault(f'pyhf.optimize.{import_name}', None)
    CACHE_MODULE, sys.modules[f'pyhf.optimize.{import_name}'] = (
        sys.modules[f'pyhf.optimize.{import_name}'],
        None,
    )
    try:
        delattr(pyhf.optimize, module_name)
    except:  # noqa: E722
        pass

    with expectation:
        getattr(pyhf.optimize, module_name)

    # put back
    CACHE_BACKEND, sys.modules[backend_name] = None, CACHE_BACKEND
    CACHE_MODULE, sys.modules[f'pyhf.optimize.{import_name}'] = (
        None,
        CACHE_MODULE,
    )
