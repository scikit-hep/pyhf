import pytest
import pyhf
import sys
import tarfile
import json
import pathlib
import distutils.dir_util


# Factory as fixture pattern
@pytest.fixture
def get_json_from_tarfile():
    def _get_json_from_tarfile(archive_data_path, json_name):
        with tarfile.open(archive_data_path, "r:gz") as archive:
            json_file = (
                archive.extractfile(archive.getmember(json_name)).read().decode("utf8")
            )
        return json.loads(json_file)

    return _get_json_from_tarfile


@pytest.fixture(scope='function')
def isolate_modules():
    """
    This fixture isolates the sys.modules imported in case you need to mess around with them and do not want to break other tests.

    This is not done automatically.
    """
    CACHE_MODULES = sys.modules.copy()
    yield isolate_modules
    sys.modules.update(CACHE_MODULES)


@pytest.fixture(scope='function', autouse=True)
def reset_events():
    """
    This fixture is automatically run to clear out the events registered before and after a test function runs.
    """
    pyhf.events.__events.clear()
    pyhf.events.__disabled_events.clear()
    yield reset_events
    pyhf.events.__events.clear()
    pyhf.events.__disabled_events.clear()


@pytest.fixture(scope='function', autouse=True)
def reset_backend():
    """
    This fixture is automatically run to reset the backend before and after a test function runs.
    """
    pyhf.set_backend('numpy', default=True)
    yield reset_backend
    pyhf.set_backend('numpy', default=True)


@pytest.fixture(
    scope='function',
    params=[
        (pyhf.tensor.numpy_backend(), None),
        (pyhf.tensor.pytorch_backend(), None),
        (pyhf.tensor.pytorch_backend(precision='64b'), None),
        (pyhf.tensor.tensorflow_backend(), None),
        (pyhf.tensor.jax_backend(), None),
        (
            pyhf.tensor.numpy_backend(poisson_from_normal=True),
            pyhf.optimize.minuit_optimizer(),
        ),
    ],
    ids=['numpy', 'pytorch', 'pytorch64', 'tensorflow', 'jax', 'numpy_minuit'],
)
def backend(request):
    # a better way to get the id? all the backends we have so far for testing
    param_ids = request._fixturedef.ids
    # the backend we're using: numpy, tensorflow, etc...
    param_id = param_ids[request.param_index]
    # name of function being called (with params), the original name is .originalname
    func_name = request._pyfuncitem.name

    # skip backends if specified
    skip_backend = request.node.get_closest_marker(f'skip_{param_id}')
    # allow the specific backend to fail if specified
    fail_backend = request.node.get_closest_marker(f'fail_{param_id}')
    # only look at the specific backends
    only_backends = [
        pid for pid in param_ids if request.node.get_closest_marker(f'only_{pid}')
    ]

    if skip_backend and (param_id in only_backends):
        raise ValueError(
            f"Must specify skip_{param_id} or only_{param_id} but not both!"
        )

    if skip_backend:
        pytest.skip(f"skipping {func_name} as specified")
    elif only_backends and param_id not in only_backends:
        pytest.skip(
            f"skipping {func_name} as specified to only look at: {', '.join(only_backends)}"
        )

    if fail_backend:
        # Mark the test as xfail to actually run it and ensure that it does
        # fail. pytest.mark.xfail checks for failure, while pytest.xfail
        # assumes failure and skips running the test.
        # c.f. https://docs.pytest.org/en/6.2.x/skipping.html#xfail
        request.node.add_marker(
            pytest.mark.xfail(reason=f"expect {func_name} to fail as specified")
        )

    # actual execution here, after all checks is done
    pyhf.set_backend(*request.param)

    yield request.param


@pytest.fixture(
    scope='function',
    params=[0, 1, 2, 4],
    ids=['interpcode0', 'interpcode1', 'interpcode2', 'interpcode4'],
)
def interpcode(request):
    yield request.param


@pytest.fixture(scope='function')
def datadir(tmp_path, request):
    """
    Fixture responsible for searching a folder with the same name of test
    module and, if available, moving all contents to a temporary directory so
    tests can use them freely.
    """
    # this gets the module name (e.g. /path/to/pyhf/tests/test_schema.py)
    # and then gets the directory by removing the suffix (e.g. /path/to/pyhf/tests/test_schema)
    test_dir = pathlib.Path(request.module.__file__).with_suffix('')

    if test_dir.is_dir():
        distutils.dir_util.copy_tree(test_dir, str(tmp_path))
        # shutil is nicer, but doesn't work: https://bugs.python.org/issue20849
        # Once pyhf is Python 3.8+ only then the below can be used.
        # shutil.copytree(test_dir, tmpdir)

    return tmp_path
