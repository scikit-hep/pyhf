registry = {}

class IModifier(type):
    def __new__(cls, clsname, bases, attrs):
        global registry
        newclass = super(IModifier, cls).__new__(cls, clsname, bases, attrs)
        registry[newclass.__name__] = newclass
        return newclass
