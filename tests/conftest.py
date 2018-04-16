import pytest
import pyhf

@pytest.fixture(scope='function', autouse=True)
def reset_backend():
  yield reset_backend
  pyhf.set_backend(pyhf.default_backend)
