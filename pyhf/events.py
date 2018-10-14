import sys

# this is meant to wrap the Wrapper object
class Wrapper(object):
    __events = {}
    class Events(list):
        def __call__(self, *args, **kwargs):
            for f in self:
                f(*args, **kwargs)

        def __repr__(self):
            return "Events(%s)" % list.__repr__(self)

    def __init__(self, name, filepath):
        import logging
        self.log = logging.getLogger(name)
        self.__name__ = name
        self.__file__ = filepath

    """
    If we reach this, we already know the attribute is missing...
    """
    def __getattr__(self, name):
        def noop(*args, **kwargs): pass
        if name.startswith('trigger_'):
            self.log.error('Triggering "{0:s}" but nobody is listening!'.format(name.split('_')[-1]))
            return noop
        raise AttributeError

    """

        This is meant to be used as a decorator.

        >>> @pyhf.events.subscribe('myevent')
        ... def test(a,b):
        ...   print a+b
        ...
        >>> pyhf.events.trigger_myevent(1,2)
        3
    """
    def subscribe(self, event):
        def __decorator(func):
            self.__events.setdefault(event, self.Events()).append(func)
            setattr(self, 'trigger_{0:s}'.format(event), self.__events.get(event))
            return func
        return __decorator

    def unsubscribe(self, event):
        if event in self.__events:
            events = self.__events.pop(event)
            del events[:]
            delattr(self, 'trigger_{0:s}'.format(event))
        return

sys.modules[__name__] = Wrapper(__name__, __file__)
