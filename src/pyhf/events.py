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


class Callables():
    def __init__(self):
        self._callbacks = []

    def _flush(self):
        _callbacks = []
        for func, arg in self._callbacks:
            if arg is not None:
                arg = arg()
                if arg is None:
                    print(func, arg, 'None')
                    continue
            _callbacks.append((func, arg))
        self._callbacks = _callbacks

    def append(self, callback):
        try:
            # methods
            callback_ref = weakref.WeakMethod(callback), weakref.ref(callback.__self__)
        except AttributeError:
            callback_ref = weakref.ref(callback), None
        self._callbacks.append(callback_ref)

    def __call__(self, *args, **kwargs):
        self._flush()
        for func, _ in self._callbacks:
            # weakref: needs to be de-ref'd first before calling
            func()(*args, **kwargs)

    def __iter__(self):
        self._flush()
        return iter(self._callbacks)

    def __getitem__(self, index):
        self._flush()
        return self._callbacks[index]

    def __len__(self):
        self._flush()
        return len(self._callbacks)

    def __repr__(self):
        return f"Callables({self._callbacks})"


def subscribe(event):
    """
    This is meant to be used as a decorator.
    """
    # Example:
    #
    # >>> @pyhf.events.subscribe('myevent')
    # ... def test(a,b):
    # ...   print a+b
    # ...
    # >>> pyhf.events.trigger_myevent(1,2)
    # 3
    global __events

    def __decorator(func):
        __events.setdefault(event, Callables()).append(func)
        return func

    return __decorator


def register(event):
    """
    This is meant to be used as a decorator to register a function for triggering events.

    This creates two events: "<event_name>::before" and "<event_name>::after"
    """
    # Examples:
    #
    # >>> @pyhf.events.register('test_func')
    # ... def test(a,b):
    # ...   print a+b
    # ...
    # >>> @pyhf.events.subscribe('test_func::before')
    # ... def precall():
    # ...   print 'before call'
    # ...
    # >>> @pyhf.events.subscribe('test_func::after')
    # ... def postcall():
    # ...   print 'after call'
    # ...
    # >>> test(1,2)
    # "before call"
    # 3
    # "after call"
    # >>>

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
