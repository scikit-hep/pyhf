import weakref
from functools import wraps

__events = {}
__disabled_events = set()

__all__ = [
    "Callables",
    "disable",
    "enable",
    "noop",
    "register",
    "subscribe",
    "trigger",
]


def __dir__():
    return __all__


def noop(*args, **kwargs):
    pass


class Callables:
    def __init__(self):
        self._callbacks = []

    @property
    def callbacks(self):
        """
        Get the current list of living callbacks.
        """
        self._flush()
        return self._callbacks

    def append(self, callback):
        """
        Append a new bound method as a callback to the list of callables.
        """
        try:
            # methods
            callback_ref = weakref.ref(callback.__func__), weakref.ref(
                callback.__self__
            )
        except AttributeError:
            callback_ref = weakref.ref(callback), None
        self._callbacks.append(callback_ref)

    def _flush(self):
        """
        Flush the list of callbacks with those who are weakly-referencing deleted objects.

        Note: must interact with the self._callbacks directly, and not
        self.callbacks, to avoid infinite recursion.
        """
        _callbacks = []
        for func, arg in self._callbacks:
            if arg is not None:
                arg_ref = arg()
                if arg_ref is None:
                    continue
            _callbacks.append((func, arg))
        self._callbacks = _callbacks

    def __call__(self, *args, **kwargs):
        for func, arg in self.callbacks:
            # weakref: needs to be de-ref'd first before calling
            if arg is not None:
                func()(arg(), *args, **kwargs)
            else:
                func()(*args, **kwargs)

    def __iter__(self):
        return iter(self.callbacks)

    def __getitem__(self, index):
        return self.callbacks[index]

    def __len__(self):
        return len(self.callbacks)

    def __repr__(self):
        return f"Callables({self.callbacks})"


def subscribe(event):
    """
    Subscribe a function or object method as a callback to an event.

    .. note::

     This is meant to be used as a decorator.

    Args:
        event (:obj:`str`): The name of the event to subscribe to.

    Returns:
        :obj:`function`: Decorated function.

    Example:
        >>> import pyhf
        >>> @pyhf.events.subscribe("myevent")
        ... def test(a, b):
        ...     print(a + b)
        ...
        >>> pyhf.events.trigger("myevent")(1, 2)
        3

    """

    global __events

    def __decorator(func):
        __events.setdefault(event, Callables()).append(func)
        return func

    return __decorator


def register(event):
    """
    Register a function or object method to trigger an event.  This creates two
    events: ``{event_name}::before`` and ``{event_name}::after``.

    .. note::

     This is meant to be used as a decorator.

    Args:
        event (:obj:`str`): The name of the event to subscribe to.

    Returns:
        :obj:`function`: Decorated function.

    Example:
        >>> import pyhf
        >>> @pyhf.events.register("test_func")
        ... def test(a, b):
        ...     print(a + b)
        ...
        >>> @pyhf.events.subscribe("test_func::before")
        ... def precall():
        ...     print("before call")
        ...
        >>> @pyhf.events.subscribe("test_func::after")
        ... def postcall():
        ...     print("after call")
        ...
        >>> test(1, 2)
        before call
        3
        after call

    """

    def _register(func):
        @wraps(func)
        def register_wrapper(*args, **kwargs):
            trigger(f"{event:s}::before")()
            result = func(*args, **kwargs)
            trigger(f"{event:s}::after")()
            return result

        return register_wrapper

    return _register


def trigger(event):
    """
    Trigger an event if not disabled.
    """
    global __events, __disabled_events, noop
    is_noop = bool(event in __disabled_events or event not in __events)
    return noop if is_noop else __events.get(event)


def disable(event):
    """
    Disable an event from firing.
    """
    global __disabled_events
    __disabled_events.add(event)


def enable(event):
    """
    Enable an event to be fired if disabled.
    """
    global __disabled_events
    __disabled_events.remove(event)
