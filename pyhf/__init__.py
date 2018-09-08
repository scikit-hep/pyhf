from . import tensor, optimize
tensorlib = tensor.numpy_backend()
default_backend = tensorlib
optimizer = optimize.scipy_optimizer()
default_optimizer = optimizer

def get_backend():
    """
    Get the current backend and the associated optimizer

    Returns:
        backend, optimizer

    Example:
        >>> import pyhf
        >>> pyhf.get_backend()
        (<pyhf.tensor.numpy_backend.numpy_backend object at 0x...>, <pyhf.optimize.opt_scipy.scipy_optimizer object at 0x...>)

    """
    global tensorlib
    global optimizer
    return tensorlib, optimizer

def set_backend(backend):
    """
    Set the backend and the associated optimizer

    Args:
        backend: One of the supported pyhf backends: NumPy,
                 TensorFlow, PyTorch, and MXNet

    Returns:
        None

    Example:
        >>> import pyhf
        >>> import tensorflow as tf
        >>> pyhf.set_backend(pyhf.tensor.tensorflow_backend(session=tf.Session()))
    """
    global tensorlib
    global optimizer

    tensorlib = backend
    if isinstance(tensorlib, tensor.tensorflow_backend):
        optimizer = optimize.tflow_optimizer(tensorlib)
    elif isinstance(tensorlib, tensor.pytorch_backend):
        optimizer = optimize.pytorch_optimizer(tensorlib=tensorlib)
    # TODO: Add support for mxnet_optimizer()
    # elif isinstance(tensorlib, tensor.mxnet_backend):
    #     optimizer = mxnet_optimizer()
    else:
        optimizer = optimize.scipy_optimizer()


from .pdf import Model
__all__ = ["Model", "utils", "modifiers"]
