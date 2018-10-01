class Backend(object):
    """General backend class for all of our backends"""

    def __eq__(self, other):
        if isinstance(other, Backend):
            return self.__class__ == other.__class__
        return False

