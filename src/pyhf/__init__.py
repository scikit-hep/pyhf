from .tensor import BackendRetriever as tensor
from .optimize import OptimizerRetriever as optimize
from .version import __version__
from .exceptions import InvalidBackend
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
def set_backend(backend, custom_optimizer=None, _session=None):
    """
    Set the backend and the associated optimizer

    Example:
        >>> import pyhf
        >>> import tensorflow as tf
        >>> sess = tf.compat.v1.Session()
        >>> pyhf.set_backend("tensorflow", _session=sess)
        >>> pyhf.tensorlib.name
        'tensorflow'
        >>> pyhf.set_backend(b"pytorch")
        >>> pyhf.tensorlib.name
        'pytorch'
        >>> pyhf.set_backend(pyhf.tensor.numpy_backend())
        >>> pyhf.tensorlib.name
        'numpy'

    Args:
        backend (`str` or `pyhf.tensor` backend): One of the supported pyhf backends: NumPy, TensorFlow, and PyTorch
        custom_optimizer (`pyhf.optimize` optimizer): Optional custom optimizer defined by the user
        _session (|tf.compat.v1.Session|_): TensorFlow v1 compatible Session to use when the :code:`"tensorflow"` backend API is used

    .. |tf.compat.v1.Session| replace:: ``tf.compat.v1.Session``
    .. _tf.compat.v1.Session: https://www.tensorflow.org/api_docs/python/tf/compat/v1/Session

    Returns:
        None
    """
    global tensorlib
    global optimizer

    if isinstance(backend, (str, bytes)):
        if isinstance(backend, bytes):
            backend = backend.decode("utf-8")
        backend = backend.lower()
        # Needed while still using TF v1.0 API
        if backend == "tensorflow":
            backend = tensor.tensorflow_backend(session=_session)
        else:
            try:
                backend = getattr(tensor, "{0:s}_backend".format(backend))()
            except TypeError:
                raise InvalidBackend(
                    "The backend provided is not supported: {0:s}. Select from one of the supported backends: numpy, tensorflow, pytorch".format(
                        backend
                    )
                )

    _name_supported = getattr(tensor, "{0:s}_backend".format(backend.name))
    if _name_supported:
        if not isinstance(backend, _name_supported):
            raise AttributeError(
                "'{0:s}' is not a valid name attribute for backend type {1}\n                 Custom backends must have names unique from supported backends".format(
                    backend.name, type(backend)
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
