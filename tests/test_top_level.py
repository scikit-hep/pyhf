import pyhf
import pytest
import numpy as np


def test_set_backend(backend):
    tb, _ = pyhf.get_backend()
    pyhf.set_backend(tb.name)


@pytest.mark.xfail(raises=ValueError)
def test_supported_backends():
    pyhf.set_backend("fail")
