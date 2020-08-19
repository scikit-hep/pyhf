import weakref

__events = {}
__disabled_events = set([])


def noop(*args, **kwargs):
    pass


class WeakList(list):
    def append(self, item):
        list.append(self, weakref.WeakMethod(item, self.remove))


class Callables(WeakList):
    def __call__(self, *args, **kwargs):
        for func in self:
            # weakref: needs to be de-ref'd first before calling
            func()(*args, **kwargs)

    def __repr__(self):
        return "Callables(%s)" % list.__repr__(self)


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
        def register_wrapper(*args, **kwargs):
            trigger("{0:s}::before".format(event))()
            result = func(*args, **kwargs)
            trigger("{0:s}::after".format(event))()
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
