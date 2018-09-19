import pytest
import pyhf
import tensorflow as tf

@pytest.fixture(scope='function', autouse=True, params=[
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
    # a better way to get the id?
    param_id = request._fixturedef.ids[request.param_index]
    skip_backend = request.node.get_marker('skip_{0:s}'.format(param_id))
    fail_backend = request.node.get_marker('fail_{0:s}'.format(param_id))
    if skip_backend:
        pytest.skip("skipping {0:s} as specified".format(param_id))
        return
    if fail_backend:
        pytest.skip("expect {0:s} to fail as specified".format(param_id))
        return

    pyhf.set_backend(request.param)
    if isinstance(request.param, pyhf.tensor.tensorflow_backend):
        tf.reset_default_graph()
        pyhf.tensorlib.session = tf.Session()

    yield request.param

    pyhf.set_backend(pyhf.default_backend)
