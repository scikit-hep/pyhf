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
