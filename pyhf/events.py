__events = {}
__disabled_events = set([])

def noop(*args, **kwargs): pass

class Callables(list):
    def __call__(self, *args, **kwargs):
        for f in self:
            f(*args, **kwargs)

    def __repr__(self):
        return "Callables(%s)" % list.__repr__(self)

"""

    This is meant to be used as a decorator.

    >>> @pyhf.events.subscribe('myevent')
    ... def test(a,b):
    ...   print a+b
    ...
    >>> pyhf.events.trigger_myevent(1,2)
    3
"""
def subscribe(event):
    global __events
    def __decorator(func):
        __events.setdefault(event, Callables()).append(func)
        return func
    return __decorator

"""

    This is meant to be used as a decorator to register a function for triggering events.

    This creates two events: "<event_name>::before" and "<event_name>::after"

    >>> @pyhf.events.register('test_func')
    ... def test(a,b):
    ...   print a+b
    ...
    >>> @pyhf.events.subscribe('test_func::before')
    ... def precall():
    ...   print 'before call'
    ...
    >>> @pyhf.events.subscribe('test_func::after')
    ... def postcall():
    ...   print 'after call'
    ...
    >>> test(1,2)
    "before call"
    3
    "after call"
    >>>

"""
def register(event):
    def _register(func):
        def register_wrapper(*args, **kwargs):
            trigger("{0:s}::before".format(event))()
            result = func(*args, **kwargs)
            trigger("{0:s}::after".format(event))()
            return result
        return register_wrapper
    return _register

"""
Trigger an event if not disabled.
"""
def trigger(event):
    global __events, __disabled_events, noop
    is_noop = bool(event in __disabled_events or event not in __events)
    return noop if is_noop else __events.get(event)

"""
Disable an event from firing.
"""
def disable(event):
    global __disabled_events
    __disabled_events.add(event)

"""
Enable an event to be fired if disabled.
"""
def enable(event):
    global __disabled_events
    __disabled_events.remove(event)
