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
        >>> backend, optimizer = pyhf.get_backend()
        >>> backend
        <pyhf.tensor.numpy_backend.numpy_backend object at 0x...>
        >>> optimizer
        <pyhf.optimize.scipy_optimizer object at 0x...>

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
def set_backend(backend, custom_optimizer=None, precision=None):
    """
    Set the backend and the associated optimizer

    Example:
        >>> import pyhf
        >>> pyhf.set_backend("tensorflow")
        >>> pyhf.tensorlib.name
        'tensorflow'
        >>> pyhf.tensorlib.precision
        '64b'
        >>> pyhf.set_backend(b"pytorch", precision="32b")
        >>> pyhf.tensorlib.name
        'pytorch'
        >>> pyhf.tensorlib.precision
        '32b'
        >>> pyhf.set_backend(pyhf.tensor.numpy_backend())
        >>> pyhf.tensorlib.name
        'numpy'
        >>> pyhf.tensorlib.precision
        '64b'

    Args:
        backend (`str` or `pyhf.tensor` backend): One of the supported pyhf backends: NumPy, TensorFlow, PyTorch, and JAX
        custom_optimizer (`pyhf.optimize` optimizer): Optional custom optimizer defined by the user
        precision (`str`): Floating point precision to use in the backend: ``64b`` or ``32b``. Default is ``64b``.

    Returns:
        None
    """
    global tensorlib
    global optimizer

    _valid_precisions = ["32b", "64b"]
    _precision = "64b" if precision is None else precision

    if isinstance(_precision, bytes):
        _precision = _precision.decode("utf-8")
    _precision = _precision.lower()

    if isinstance(backend, (str, bytes)):
        if isinstance(backend, bytes):
            backend = backend.decode("utf-8")
        backend = backend.lower()
        try:
            backend = getattr(tensor, "{0:s}_backend".format(backend))(
                precision=_precision
            )
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
    if backend.precision not in _valid_precisions:
        raise InvalidBackend(
            f"The backend precision provided is not supported: {backend.precision:s}. Select from one of the supported precisions: {', '.join([str(v) for v in _valid_precisions])}"
        )
    # If "precision" arg passed, it should always win
    # If no "precision" arg, defer to tensor backend object API if set there
    if precision is not None:
        if backend.precision != precision:
            backend = getattr(tensor, "{0:s}_backend".format(backend.name))(
                precision=_precision
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
                        f"'{custom_optimizer.name}' is not a valid name attribute for optimizer type {type(custom_optimizer)}\n                 Custom optimizers must have names unique from supported optimizers"
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
