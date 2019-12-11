import pyhf
import pytest
import numpy as np


def test_set_backend_by_name(backend):
    tb, _ = pyhf.get_backend()
    pyhf.set_backend(tb.name)
    assert isinstance(
        pyhf.tensorlib, getattr(pyhf.tensor, "{0:s}_backend".format(tb.name))
    )


@pytest.mark.parametrize("backend_name", ["numpy", "tensorflow", "pytorch"])
def test_set_backend_by_string(backend_name):
    pyhf.set_backend(backend_name)
    assert isinstance(
        pyhf.tensorlib, getattr(pyhf.tensor, "{0:s}_backend".format(backend_name))
    )


@pytest.mark.parametrize("backend_name", [b"numpy", b"tensorflow", b"pytorch"])
def test_set_backend_by_bytestring(backend_name):
    pyhf.set_backend(backend_name)
    assert isinstance(
        pyhf.tensorlib,
        getattr(pyhf.tensor, "{0:s}_backend".format(backend_name.decode("utf-8"))),
    )


def test_supported_backends():
    with pytest.raises(pyhf.exceptions.InvalidBackend):
        pyhf.set_backend("fail")
    with pytest.raises(pyhf.exceptions.InvalidBackend):
        pyhf.set_backend(b"fail")
