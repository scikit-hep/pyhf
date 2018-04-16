from six import string_types
import logging
log = logging.getLogger(__name__)

registry = {}

class InvalidModifier(Exception):
  pass

'''
Check if given object contains the right structure for constrained and unconstrained modifiers
'''
def validate_modifier_structure(modifier, constrained):
    required_methods = ['__init__', 'add_sample']
    required_constrained_methods = ['alphas', 'pdf', 'expected_data']

    for method in required_methods + required_constrained_methods*constrained:
        if not hasattr(modifier, method):
          raise InvalidModifier('Expected {0:s} method on {1:s}constrained modifier {2:s}'.format(method, '' if constrained else 'un', modifier.__name__))
    return True

'''
Consistent add_to_registry() function that handles actually adding thing to the registry.

Raises an error if the name to register for the modifier already exists in the registry,
or if the modifier does not have the right structure.
'''
def add_to_registry(cls, cls_name=None, constrained=False):
  global registry
  cls_name = cls_name if cls_name else cls.__name__
  if cls_name in registry: raise KeyError('The modifier name "{0:s}" is already taken.'.format(cls_name))
  # validate the structure
  validate_modifier_structure(cls, constrained)
  # set is_constrained
  cls.is_constrained = constrained
  registry[cls_name] = cls

'''
Decorator for registering modifiers. To flag the modifier as a constrained modifier, add `constrained=True`.

  >>> @modifiers.modifier
  >>> ... class myCustomModifier(object):
  >>> ...   def __init__(self): pass
  >>> ...   def add_sample(self): pass
  >>>
  >>> @modifiers.modifier(name='myCustomNamer')
  >>> ... class myCustomModifier(object):
  >>> ...   def __init__(self): pass
  >>> ...   def add_sample(self): pass
  >>>
  >>> @modifiers.modifier(constrained=True)
  >>> ... class myCustomModifier(object):
  >>> ...   def __init__(self): pass
  >>> ...   def add_sample(self): pass
  >>> ...   def alphas(self): pass
  >>> ...   def pdf(self): pass
  >>> ...   def expected_data(self): pass

Should raise error if not passed in one argument (the class to wrap for automatically grabbing the class name; or the string to name the class).

  >>> @modifiers.modifier(name='myCustomNamer')
  >>> ... class myCustomModifier(object):
  >>> ...   def __init__(self): pass
  >>> ...   def add_sample(self): pass
  >>>
  InvalidModifier: Expected alphas method on constrained modifier myCustomModifier

'''
def modifier(*args, **kwargs):
    name = kwargs.pop('name', None)
    constrained = bool(kwargs.pop('constrained', False))
    # check for unparsed keyword arguments
    if len(kwargs) != 0:
        raise ValueError('Unparsed keyword arguments {}'.format(kwargs.keys()))
    # check to make sure the given name is a string, if passed in one
    if not isinstance(name, string_types) and name is not None:
        raise TypeError('@modifier must be given a string. You gave it {}'.format(type(name)))

    def _modifier(name, constrained):
        def wrapper(cls):
            add_to_registry(cls, cls_name=name, constrained=constrained)
            return cls
        return wrapper

    if len(args) == 0:
        # called like @modifier(name='foo', constrained=False)
        return _modifier(name, constrained)
    elif len(args) == 1:
        # called like @modifier
        if not callable(args[0]):
            raise ValueError('You must decorate a callable python object')
        add_to_registry(args[0], cls_name=name, constrained=constrained)
        return args[0]
    else:
        raise ValueError('@modifier must be called with only keyword arguments, @modifier(name=\'foo\'), or no arguments, @modifier; ({0:d} given)'.format(len(args)))

from .histosys import histosys
from .normfactor import normfactor
from .normsys import normsys
from .shapefactor import shapefactor
from .shapesys import shapesys
__all__ = [histosys,normfactor,normsys,shapefactor,shapesys]
