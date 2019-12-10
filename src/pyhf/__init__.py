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
def set_backend(backend, custom_optimizer=None, custom_backend=False, _session=None):
    """
    Set the backend and the associated optimizer

    Example:
        >>> import pyhf
        >>> import tensorflow as tf
        >>> sess = tf.compat.v1.Session()
        >>> pyhf.set_backend("tensorflow", _session=sess)

    Args:
        backend (`str` or `pyhf.tensor` backend): One of the supported pyhf backends: NumPy, TensorFlow, and PyTorch
        custom_optimizer (`pyhf.optimize` optimizer): Optional custom optimizer defined by the user
        custom_backend (`bool`): If :code:`True` the :code:`backend` argument is allowed to be a custom user defined backend outside of the supported backends
        _session (|tf.compat.v1.Session|_): TensorFlow v1 compatible Session to use when the :code:`"tensorflow"` backend API is used

    .. |tf.compat.v1.Session| replace:: ``tf.compat.v1.Session``
    .. _tf.compat.v1.Session: https://www.tensorflow.org/api_docs/python/tf/compat/v1/Session

    Returns:
        None
    """
    global tensorlib
    global optimizer

    supported_backend_types = (
        tensor.numpy_backend,
        tensor.tensorflow_backend,
        tensor.pytorch_backend,
    )
    supported_backend_names = (backend().name for backend in supported_backend_types)

    if isinstance(backend, str) and (
        backend in supported_backend_names or backend == "numpy_minuit"
    ):
        if backend == "numpy":
            backend = tensor.numpy_backend()
        elif backend == "tensorflow":
            backend = tensor.tensorflow_backend(session=_session)
        elif backend == "pytorch":
            backend = tensor.pytorch_backend()
        elif backend == "numpy_minuit":
            backend = tensor.numpy_backend(poisson_from_normal=True)
    elif not isinstance(backend, supported_backend_types) and not custom_backend:
        raise ValueError(
            "'{0:s}' is not a supported backend.\n             Select from one of the supported backends: numpy, tensorflow, pytorch".format(
                backend
            )
        )

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
from . import infer

__all__ = [
    'Model',
    'Workspace',
    'infer',
    'utils',
    'modifiers',
    'simplemodels',
    '__version__',
]
