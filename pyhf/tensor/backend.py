import inspect
from six import string_types


class Backend(object):
    """General backend class for all of our backends"""

    def __eq__(self, other):
        if isinstance(other, Backend):
            # if this is comparing to Backend objects
            return self.__class__ == other.__class__
        elif inspect.isclass(other):
            # if this is comparing a Backend class
            return self.__class__ == other
        elif isinstance(other, string_types):
            # if this is comparing to a string like 'numpy'
            return self.name == other
        else:
            return False
        return False # always return false
