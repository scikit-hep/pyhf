import pyhf
import pytest
import numpy as np


def test_set_backend(backend):
    tb, _ = pyhf.get_backend()
    pyhf.set_backend(tb.name)


@pytest.mark.parametrize(
    "backend_name", ["numpy", "tensorflow", "pytorch", "numpy_minuit"]
)
def test_set_backend_by_string(backend_name):
    pyhf.set_backend(backend_name)


@pytest.mark.xfail(raises=ValueError)
def test_supported_backends():
    pyhf.set_backend("fail")
