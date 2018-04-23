import pytest
import pyhf

@pytest.fixture(scope='function', autouse=True)
def reset_backend():
  yield reset_backend
  pyhf.set_backend(pyhf.default_backend)

import pytest
def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true", default=False, help="run slow tests")

def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
