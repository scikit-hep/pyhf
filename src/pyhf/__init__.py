from .tensor import BackendRetriever as tensor
from .optimize import OptimizerRetriever as optimize
from .version import __version__
from . import events

tensorlib = tensor.numpy_backend()
default_backend = tensorlib
optimizer = optimize.scipy_optimizer()
default_optimizer = optimizer


def get_backend():
    """
    Get the current backend and the associated optimizer

    Example:
        >>> import pyhf
        >>> pyhf.get_backend()
        (<pyhf.tensor.numpy_backend.numpy_backend object at 0x...>, <pyhf.optimize.opt_scipy.scipy_optimizer object at 0x...>)

    Returns:
        backend, optimizer
    """
    global tensorlib
    global optimizer
    return tensorlib, optimizer


@events.register('change_backend')
def set_backend(backend, custom_optimizer=None):
    """
    Set the backend and the associated optimizer

    Example:
        >>> import pyhf
        >>> import tensorflow as tf
        >>> pyhf.set_backend(pyhf.tensor.tensorflow_backend(session=tf.Session()))

    Args:
        backend: One of the supported pyhf backends: NumPy, TensorFlow, and PyTorch

    Returns:
        None
    """
    global tensorlib
    global optimizer

    # need to determine if the tensorlib changed or the optimizer changed for events
    tensorlib_changed = bool(backend.name != tensorlib.name)
    optimizer_changed = False

    if backend.name == 'tensorflow':
        new_optimizer = (
            custom_optimizer if custom_optimizer else optimize.tflow_optimizer(backend)
        )
        if tensorlib.name == 'tensorflow':
            tensorlib_changed |= bool(backend.session != tensorlib.session)
    elif backend.name == 'pytorch':
        new_optimizer = (
            custom_optimizer
            if custom_optimizer
            else optimize.pytorch_optimizer(tensorlib=backend)
        )
    else:
        new_optimizer = (
            custom_optimizer if custom_optimizer else optimize.scipy_optimizer()
        )

    optimizer_changed = bool(optimizer != new_optimizer)
    # set new backend
    tensorlib = backend
    optimizer = new_optimizer
    # trigger events
    if tensorlib_changed:
        events.trigger("tensorlib_changed")()
    if optimizer_changed:
        events.trigger("optimizer_changed")()


from .pdf import Model
from .workspace import Workspace
from . import simplemodels

__all__ = ['Model', 'Workspace', 'utils', 'modifiers', 'simplemodels', '__version__']
