registry = {}

'''
Consistent add_to_registry() function that handles actually adding thing to the registry.

Raises an error if the name to register for the modifier already exists in the registry.
'''
def add_to_registry(cls, cls_name=None):
  global registry
  cls_name = cls_name if cls_name else cls.__name__
  if cls_name in registry: raise KeyError('The modifier name "{0:s}" is already taken.'.format(cls_name))
  registry[cls_name] = cls

'''
Decorator for registering modifiers. Two ways to use it.

Way 1: automatically determine the name of the modifier using cls.__name__

  >>> @modifiers.modifier
  >>> ... class histosys(object):
  >>> ...   pass

Way 2: pass in a name to use for the modifier

  >>> @modifiers.modifier('mymodifier')
  >>> ... class histosys(object):
  >>> ...   pass
  >>> ...

Should raise error if not passed in one argument (the class to wrap for automatically grabbing the class name; or the string to name the class).

'''
def modifier(*args):
    if len(args) != 1:
        raise ValueError('@modifier takes exactly 1 argument ({0:d} given)'.format(len(args)))

    def _modifier(name):
        def wrapper(cls):
            add_to_registry(cls, name)
            return cls
        return wrapper

    if callable(args[0]):
        add_to_registry(args[0])
        return args[0]
    elif isinstance(args[0], basestring):
        return _modifier(args[0])
    else:
        raise TypeError('@modifier must be given a basestring instance (string, unicode). You gave it {}'.format(type(args[0])))

from .histosys import histosys
from .normfactor import normfactor
from .normsys import normsys
from .shapefactor import shapefactor
from .shapesys import shapesys
__all__ = [histosys,normfactor,normsys,shapefactor,shapesys]

