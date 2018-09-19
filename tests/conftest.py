import pytest
import pyhf
import tensorflow as tf

@pytest.fixture(scope='function', params=[
                             pyhf.tensor.numpy_backend(),
                             pyhf.tensor.tensorflow_backend(session=tf.Session()),
                             pyhf.tensor.pytorch_backend(),
                             pyhf.tensor.mxnet_backend(),
                         ],
                         ids=[
                             'numpy',
                             'tensorflow',
                             'pytorch',
                             'mxnet',
                         ])
def backend(request):
    param = request.param
    # a better way to get the id? all the backends we have so far for testing
    param_ids = request._fixturedef.ids
    # the backend we're using: numpy, tensorflow, etc...
    param_id = param_ids[request.param_index]
    # name of function being called (with params), the original name is .originalname
    func_name = request._pyfuncitem.name

    # skip backends if specified
    skip_backend = request.node.get_marker('skip_{param}'.format(param=param_id))
    # allow the specific backend to fail if specified
    fail_backend = request.node.get_marker('fail_{param}'.format(param=param_id))
    # only look at the specific backends
    only_backends = [pid for pid in param_ids if request.node.get_marker('only_{param}'.format(param=pid))]

    if(skip_backend and (param_id in only_backends)):
        raise ValueError("Must specify skip_{param} or only_{param} but not both!".format(param=pid))

    if skip_backend:
        pytest.skip("skipping {func} as specified".format(func=func_name))
    elif only_backends and param_id not in only_backends:
        pytest.skip("skipping {func} as specified to only look at: {backends}".format(func=func_name, backends=', '.join(only_backends)))

    if fail_backend:
        pytest.xfail("expect {func} to fail as specified".format(func=func_name))

    # actual execution here, after all checks is done
    pyhf.set_backend(request.param)
    if isinstance(request.param, pyhf.tensor.tensorflow_backend):
        tf.reset_default_graph()
        pyhf.tensorlib.session = tf.Session()

    yield request.param

    pyhf.set_backend(pyhf.default_backend)
