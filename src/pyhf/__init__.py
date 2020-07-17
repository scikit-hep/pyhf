from .tensor import BackendRetriever as tensor
from .optimize import OptimizerRetriever as optimize
from .version import __version__
from .exceptions import InvalidBackend, InvalidOptimizer
from . import events

tensorlib = None
optimizer = None


def get_backend():
    """
    Get the current backend and the associated optimizer

    Example:
        >>> import pyhf
        >>> pyhf.get_backend()
        (<pyhf.tensor.numpy_backend.numpy_backend object at 0x...>, <pyhf.optimize.scipy_optimizer object at 0x...>)

    Returns:
        backend, optimizer
    """
    global tensorlib
    global optimizer
    return tensorlib, optimizer


tensorlib = tensor.numpy_backend()
default_backend = tensorlib
optimizer = optimize.scipy_optimizer()
default_optimizer = optimizer


@events.register('change_backend')
def set_backend(backend, custom_optimizer=None):
    """
    Set the backend and the associated optimizer

    Example:
        >>> import pyhf
        >>> pyhf.set_backend("tensorflow")
        >>> pyhf.tensorlib.name
        'tensorflow'
        >>> pyhf.set_backend(b"pytorch")
        >>> pyhf.tensorlib.name
        'pytorch'
        >>> pyhf.set_backend(pyhf.tensor.numpy_backend())
        >>> pyhf.tensorlib.name
        'numpy'

    Args:
        backend (`str` or `pyhf.tensor` backend): One of the supported pyhf backends: NumPy, TensorFlow, PyTorch, and JAX
        custom_optimizer (`pyhf.optimize` optimizer): Optional custom optimizer defined by the user

    Returns:
        None
    """
    global tensorlib
    global optimizer

    if isinstance(backend, (str, bytes)):
        if isinstance(backend, bytes):
            backend = backend.decode("utf-8")
        backend = backend.lower()
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
    tensorlib_changed = bool(
        (backend.name != tensorlib.name) | (backend.precision != tensorlib.precision)
    )
    optimizer_changed = False

    if custom_optimizer:
        if isinstance(custom_optimizer, (str, bytes)):
            if isinstance(custom_optimizer, bytes):
                custom_optimizer = custom_optimizer.decode("utf-8")
            try:
                new_optimizer = getattr(
                    optimize, f"{custom_optimizer.lower()}_optimizer"
                )()
            except TypeError:
                raise InvalidOptimizer(
                    f"The optimizer provided is not supported: {custom_optimizer}. Select from one of the supported optimizers: scipy, minuit"
                )
        else:
            _name_supported = getattr(
                optimize, "{0:s}_optimizer".format(custom_optimizer.name)
            )
            if _name_supported:
                if not isinstance(custom_optimizer, _name_supported):
                    raise AttributeError(
                        f"'{custom_optimizer.name}' is not a valid name attribute for optimizer type {type(custom_optimizer)}\n                 Custom backends must have names unique from supported backends"
                    )
            new_optimizer = custom_optimizer

    else:
        new_optimizer = optimize.scipy_optimizer()

    optimizer_changed = bool(optimizer != new_optimizer)
    # set new backend
    tensorlib = backend
    optimizer = new_optimizer
    # trigger events
    if tensorlib_changed:
        events.trigger("tensorlib_changed")()
    if optimizer_changed:
        events.trigger("optimizer_changed")()
    # set up any other globals for backend
    tensorlib._setup()


from .pdf import Model
from .workspace import Workspace
from . import simplemodels
from . import infer
from .patchset import PatchSet

__all__ = [
    'Model',
    'Workspace',
    'PatchSet',
    'infer',
    'utils',
    'modifiers',
    'simplemodels',
    '__version__',
]
