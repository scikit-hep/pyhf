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
            pytest.raises(pyhf.exceptions.MissingLibraries),
        ],
        [
            "tensorflow",
            "tensorflow_backend",
            "tensorflow_backend",
            pytest.raises(pyhf.exceptions.MissingLibraries),
        ],
        [
            "mxnet",
            "mxnet_backend",
            "mxnet_backend",
            pytest.raises(pyhf.exceptions.MissingLibraries),
        ],
        [
            None,
            "fake_backend",
            "fake_backend",
            pytest.raises(pyhf.exceptions.InvalidBackend),
        ],
    ],
    ids=["numpy", "pytorch", "tensorflow", "mxnet", "fake"],
)
def test_missing_backends(isolate_modules, param):
    backend_name, module_name, import_name, expectation = param

    # hide if defined
    if backend_name:
        CACHE_BACKEND, sys.modules[backend_name] = sys.modules[backend_name], None
        sys.modules.setdefault('pyhf.tensor.{}'.format(import_name), None)
        CACHE_MODULE, sys.modules['pyhf.tensor.{}'.format(module_name)] = (
            sys.modules['pyhf.tensor.{}'.format(module_name)],
            None,
        )
        try:
            delattr(pyhf.tensor, module_name)
        except:
            pass

    with expectation:
        getattr(pyhf.tensor, module_name)

    # put back
    if backend_name:
        CACHE_BACKEND, sys.modules[backend_name] = None, CACHE_BACKEND
        CACHE_MODULE, sys.modules['pyhf.tensor.{}'.format(module_name)] = (
            None,
            CACHE_MODULE,
        )


@pytest.mark.parametrize(
    "param",
    [
        ["scipy", "scipy_optimizer", "opt_scipy", pytest.raises(ImportError)],
        [
            "torch",
            "pytorch_optimizer",
            "opt_pytorch",
            pytest.raises(pyhf.exceptions.MissingLibraries),
        ],
        [
            "tensorflow",
            "tflow_optimizer",
            "opt_tflow",
            pytest.raises(pyhf.exceptions.MissingLibraries),
        ],
        [
            "iminuit",
            "minuit_optimizer",
            "opt_minuit",
            pytest.raises(pyhf.exceptions.MissingLibraries),
        ],
        [
            None,
            "fake_optimizer",
            "fake_opt",
            pytest.raises(pyhf.exceptions.InvalidOptimizer),
        ],
    ],
    ids=["scipy", "pytorch", "tensorflow", "minuit", "fake"],
)
def test_missing_optimizer(isolate_modules, param):
    backend_name, module_name, import_name, expectation = param

    # hide if defined
    if backend_name:
        CACHE_BACKEND, sys.modules[backend_name] = sys.modules[backend_name], None
        sys.modules.setdefault('pyhf.optimize.{}'.format(import_name), None)
        CACHE_MODULE, sys.modules['pyhf.optimize.{}'.format(import_name)] = (
            sys.modules['pyhf.optimize.{}'.format(import_name)],
            None,
        )
        try:
            delattr(pyhf.optimize, module_name)
        except:
            pass

    with expectation:
        getattr(pyhf.optimize, module_name)

    # put back
    if backend_name:
        CACHE_BACKEND, sys.modules[backend_name] = None, CACHE_BACKEND
        CACHE_MODULE, sys.modules['pyhf.optimize.{}'.format(import_name)] = (
            None,
            CACHE_MODULE,
        )
