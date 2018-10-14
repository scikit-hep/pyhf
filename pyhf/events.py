import logging
log = logging.getLogger(__name__)

__events = {}
__disabled_events = set([])
class Events(list):
    def __call__(self, *args, **kwargs):
        for f in self:
            f(*args, **kwargs)

    def __repr__(self):
        return "Events(%s)" % list.__repr__(self)

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
        __events.setdefault(event, Events()).append(func)
        return func
    return __decorator

"""
Trigger an event if not disabled.
"""
def trigger(event):
    global __events, __disabled_events
    def noop(*args, **kwargs): pass
    def _trigger(*args, **kwargs):
        return __events.get(event, noop)(*args, **kwargs)
    return noop if event in __disabled_events else _trigger

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
