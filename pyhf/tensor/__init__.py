from .numpy_backend import numpy_backend
assert numpy_backend

try:
    from .pytorch_backend import pytorch_backend
    assert pytorch_backend
except ImportError:
    pass

try:
    from .tensorflow_backend import tensorflow_backend
    assert tensorflow_backend
except ImportError:
    pass

try:
    from .mxnet_backend import mxnet_backend
    assert mxnet_backend
except ImportError:
    pass
