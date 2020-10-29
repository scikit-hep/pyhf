"""
Defines a synchronous Python-like Executor for pyhf: :class:`TrivialExecutor`
"""
from concurrent import futures


class TrivialExecutor(futures.Executor):
    """
    Formally satisfies the interface for a :class:`concurrent.futures.Executor`
    but the :func:`TrivialExecutor.submit` method computes its ``task``
    synchronously.
    """

    def __repr__(self):
        """ Representation of the object """
        module = type(self).__module__
        qualname = type(self).__qualname__
        return f"<{module}.{qualname} object at {hex(id(self))}>"

    def submit(self, task, *args, **kwargs):
        """
        Immediately runs ``task(*args)``.
        """
        _f = futures.Future()
        _f.set_running_or_notify_cancel()
        try:
            result = task(*args)
        except BaseException as exc:
            _f.set_exception(exc)
        else:
            _f.set_result(result)
        return _f
