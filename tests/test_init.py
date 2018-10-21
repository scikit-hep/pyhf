import pytest
import sys


@pytest.mark.parametrize(
    "param",
    [
        ["numpy", "numpy_backend", pytest.raises(ImportError)],
        ["torch", "pytorch_backend", pytest.raises(AttributeError)],
        ["tensorflow", "tensorflow_backend", pytest.raises(AttributeError)],
        ["mxnet", "mxnet_backend", pytest.raises(AttributeError)],
    ],
    ids=["numpy", "pytorch", "tensorflow", "mxnet"],
)
def test_missing_backends(isolate_modules, param):
    backend_name, module_name, expectation = param

    # delete all of pyhf to force a reload
    for k in [k for k in sys.modules.keys() if 'pyhf' in k]:
        del sys.modules[k]

    # hide
    CACHE_BACKEND, sys.modules[backend_name] = sys.modules[backend_name], None

    with expectation:
        import pyhf.tensor

        getattr(pyhf.tensor, module_name)

    # put back
    CACHE_BACKEND, sys.modules[backend_name] = None, CACHE_BACKEND
