from __future__ import annotations

import sys

from pyhf import events, exceptions
from pyhf.optimize import OptimizerRetriever
from pyhf.tensor import BackendRetriever
from pyhf.typing import Optimizer, Protocol, TensorBackend, TypedDict


class State(TypedDict):
    default: tuple[TensorBackend, Optimizer]
    current: tuple[TensorBackend, Optimizer]


class HasState(Protocol):
    state: State


this: HasState = sys.modules[__name__]
this.state = {
    'default': (None, None),  # type: ignore[typeddict-item]
    'current': (None, None),  # type: ignore[typeddict-item]
}


def get_backend(default: bool = False) -> tuple[TensorBackend, Optimizer]:
    """
    Get the current backend and the associated optimizer

    Example:
        >>> import pyhf
        >>> pyhf.set_backend("numpy")
        >>> backend, optimizer = pyhf.get_backend()
        >>> backend
        <pyhf.tensor.numpy_backend.numpy_backend object at 0x...>
        >>> optimizer
        <pyhf.optimize.scipy_optimizer object at 0x...>

    Args:
        default (:obj:`bool`): Return the default backend or not

    Returns:
        backend, optimizer
    """
    return this.state["default"] if default else this.state["current"]


_default_backend: TensorBackend = BackendRetriever.numpy_backend()
_default_optimizer: Optimizer = OptimizerRetriever.scipy_optimizer()  # type: ignore[no-untyped-call]

this.state['default'] = (_default_backend, _default_optimizer)
this.state['current'] = this.state['default']


@events.register('change_backend')
def set_backend(
    backend: str | bytes | TensorBackend,
    custom_optimizer: str | bytes | Optimizer | None = None,
    precision: str | bytes | None = None,
    default: bool = False,
) -> None:
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
        backend (:obj:`str` or :obj:`bytes` or `pyhf.tensor` backend): One of the supported pyhf backends: NumPy, TensorFlow, PyTorch, and JAX
        custom_optimizer (:obj:`str` or :obj:`bytes` or `pyhf.optimize` optimizer or :obj:`None`): Optional custom optimizer defined by the user
        precision (:obj:`str` or :obj:`bytes` or :obj:`None`): Floating point precision to use in the backend: ``64b`` or ``32b``. Default is backend dependent.
        default (:obj:`bool`): Set the backend as the default backend additionally

    Returns:
        None
    """
    _supported_precisions = ["32b", "64b"]
    backend_kwargs = {}

    if precision:
        if isinstance(precision, bytes):
            precision = precision.decode("utf-8")

        precision = precision.lower()
        if precision not in _supported_precisions:
            raise exceptions.Unsupported(
                f"The backend precision provided is not supported: {precision:s}. Select from one of the supported precisions: {', '.join([str(v) for v in _supported_precisions])}"
            )

        backend_kwargs["precision"] = precision

    if isinstance(backend, bytes):
        backend = backend.decode("utf-8")

    if isinstance(backend, str):
        backend = backend.lower()

        try:
            new_backend: TensorBackend = getattr(
                BackendRetriever, f"{backend:s}_backend"
            )(**backend_kwargs)
        except TypeError:
            raise exceptions.InvalidBackend(
                f"The backend provided is not supported: {backend:s}. Select from one of the supported backends: numpy, tensorflow, pytorch"
            )
    else:
        new_backend = backend

    _name_supported = getattr(BackendRetriever, f"{new_backend.name:s}_backend")
    if _name_supported:
        if not isinstance(new_backend, _name_supported):
            raise AttributeError(
                f"'{new_backend.name:s}' is not a valid name attribute for backend type {type(new_backend)}\n                 Custom backends must have names unique from supported backends"
            )

    # If "precision" arg passed, it should always win
    # If no "precision" arg, defer to tensor backend object API if set there
    if precision is not None and new_backend.precision != precision:
        new_backend = getattr(BackendRetriever, f"{new_backend.name:s}_backend")(
            **backend_kwargs
        )

    if custom_optimizer is None:
        new_optimizer: Optimizer = OptimizerRetriever.scipy_optimizer()  # type: ignore[no-untyped-call]
    else:
        if isinstance(custom_optimizer, bytes):
            custom_optimizer = custom_optimizer.decode("utf-8")

        if isinstance(custom_optimizer, str):
            custom_optimizer = custom_optimizer.lower()

            try:
                new_optimizer = getattr(
                    OptimizerRetriever, f"{custom_optimizer.lower()}_optimizer"
                )()
            except TypeError:
                raise exceptions.InvalidOptimizer(
                    f"The optimizer provided is not supported: {custom_optimizer}. Select from one of the supported optimizers: scipy, minuit"
                )
        else:
            new_optimizer = custom_optimizer

        _name_supported = getattr(
            OptimizerRetriever, f"{new_optimizer.name:s}_optimizer"
        )
        if _name_supported:
            if not isinstance(new_optimizer, _name_supported):
                raise AttributeError(
                    f"'{new_optimizer.name}' is not a valid name attribute for optimizer type {type(new_optimizer)}\n                 Custom optimizers must have names unique from supported optimizers"
                )

    # need to determine if the tensorlib changed or the optimizer changed for events
    tensorlib_changed = bool(
        (new_backend.name != this.state['current'][0].name)
        | (new_backend.precision != this.state['current'][0].precision)
    )
    optimizer_changed = bool(this.state['current'][1] != new_optimizer)
    # set new backend
    this.state['current'] = (new_backend, new_optimizer)
    if default:
        default_tensorlib_changed = bool(
            (new_backend.name != this.state['default'][0].name)
            | (new_backend.precision != this.state['default'][0].precision)
        )
        default_optimizer_changed = bool(this.state['default'][1] != new_optimizer)
        # trigger events
        if default_tensorlib_changed:
            events.trigger("default_tensorlib_changed")()
        if default_optimizer_changed:
            events.trigger("default_optimizer_changed")()

        this.state['default'] = this.state['current']

    # trigger events
    if tensorlib_changed:
        events.trigger("tensorlib_changed")()
    if optimizer_changed:
        events.trigger("optimizer_changed")()
    # set up any other globals for backend
    new_backend._setup()
