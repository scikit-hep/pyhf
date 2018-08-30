from six import string_types
import logging
log = logging.getLogger(__name__)

from .. import exceptions
from .. import get_backend

registry = {}

'''
Check if given object contains the right structure for constrained and unconstrained modifiers
'''
def validate_modifier_structure(modifier, constrained):
    required_methods = ['__init__', 'add_sample', 'apply']
    required_constrained_methods = ['alphas', 'pdf', 'expected_data']

    for method in required_methods + required_constrained_methods*constrained:
        if not hasattr(modifier, method):
            raise exceptions.InvalidModifier('Expected {0:s} method on {1:s}constrained modifier {2:s}'.format(method, '' if constrained else 'un', modifier.__name__))
    return True

'''
Consistent add_to_registry() function that handles actually adding thing to the registry.

Raises an error if the name to register for the modifier already exists in the registry,
or if the modifier does not have the right structure.
'''
def add_to_registry(cls, cls_name=None, constrained=False, shared=False, pdf_type='normal'):
    global registry
    cls_name = cls_name or cls.__name__
    if cls_name in registry: raise KeyError('The modifier name "{0:s}" is already taken.'.format(cls_name))
    # validate the structure
    validate_modifier_structure(cls, constrained)
    # set is_constrained
    cls.is_constrained = constrained
    cls.is_shared = shared
    if constrained:
        tensorlib, _ = get_backend()
        if not hasattr(tensorlib, pdf_type):
            raise exceptions.InvalidModifier('The specified pdf_type "{0:s}" is not valid for {1:s}({2:s}). See pyhf.tensor documentation for available pdfs.'.format(pdf_type, cls_name, cls.__name__))
        cls.pdf_type = pdf_type
    else:
        cls.pdf_type = None
    registry[cls_name] = cls

'''
Decorator for registering modifiers. To flag the modifier as a constrained modifier, add `constrained=True`.


Args:
    name: the name of the modifier to use. Use the class name by default. (default: None)
    constrained: whether the modifier is constrained or not. (default: False)
    shared: whether the modifier is shared or not. (default: False)
    pdf_type: the name of the pdf to use from tensorlib if constrained. (default: normal)

Returns:
    modifier

Raises:
    ValueError: too many keyword arguments, or too many arguments, or wrong arguments
    TypeError: provided name is not a string
    pyhf.exceptions.InvalidModifier: object does not have necessary modifier structure

Examples:

  >>> @modifiers.modifier
  >>> ... class myCustomModifier(object):
  >>> ...   def __init__(self): pass
  >>> ...   def add_sample(self): pass
  >>> ...   def apply(self): pass

  >>> @modifiers.modifier(name='myCustomNamer')
  >>> ... class myCustomModifier(object):
  >>> ...   def __init__(self): pass
  >>> ...   def add_sample(self): pass
  >>> ...   def apply(self): pass

  >>> @modifiers.modifier(shared=True)
  >>> ... class myCustomSharedModifier(object):
  >>> ...   def __init__(self): pass
  >>> ...   def add_sample(self): pass
  >>> ...   def apply(self): pass

  >>> @modifiers.modifier(constrained=False)
  >>> ... class myUnconstrainedModifier(object):
  >>> ...   def __init__(self): pass
  >>> ...   def add_sample(self): pass
  >>> ...   def apply(self): pass
  >>> ...
  >>> myUnconstrainedModifier.pdf_type
  None

  >>> @modifiers.modifier(constrained=True, pdf_type='poisson')
  >>> ... class myConstrainedCustomPoissonModifier(object):
  >>> ...   def __init__(self): pass
  >>> ...   def add_sample(self): pass
  >>> ...   def apply(self): pass
  >>> ...
  >>> myConstrainedCustomGaussianModifier.pdf_type
  'poisson'

  >>> @modifiers.modifier(constrained=True)
  >>> ... class myCustomModifier(object):
  >>> ...   def __init__(self): pass
  >>> ...   def add_sample(self): pass
  >>> ...   def apply(self): pass
  >>> ...   def alphas(self): pass
  >>> ...   def pdf(self): pass
  >>> ...   def expected_data(self): pass

  >>> @modifiers.modifier(name='myCustomNamer')
  >>> ... class myCustomModifier(object):
  >>> ...   def __init__(self): pass
  >>> ...   def add_sample(self): pass
  >>> ...   def apply(self): pass
  >>> ...
  pyhf.exceptions.InvalidModifier: Expected alphas method on constrained modifier myCustomModifier
'''
def modifier(*args, **kwargs):
    name = kwargs.pop('name', None)
    constrained = bool(kwargs.pop('constrained', False))
    shared = bool(kwargs.pop('shared', False))
    pdf_type = str(kwargs.pop('pdf_type', 'normal'))
    # check for unparsed keyword arguments
    if len(kwargs) != 0:
        raise ValueError('Unparsed keyword arguments {}'.format(kwargs.keys()))
    # check to make sure the given name is a string, if passed in one
    if not isinstance(name, string_types) and name is not None:
        raise TypeError('@modifier must be given a string. You gave it {}'.format(type(name)))

    def _modifier(name, constrained, shared, pdf_type):
        def wrapper(cls):
            add_to_registry(cls, cls_name=name, constrained=constrained, shared=shared, pdf_type=pdf_type)
            return cls
        return wrapper

    if len(args) == 0:
        # called like @modifier(name='foo', constrained=False, shared=False, pdf_type='normal')
        return _modifier(name, constrained, shared, pdf_type)
    elif len(args) == 1:
        # called like @modifier
        if not callable(args[0]):
            raise ValueError('You must decorate a callable python object')
        add_to_registry(args[0], cls_name=name, constrained=constrained, shared=shared, pdf_type=pdf_type)
        return args[0]
    else:
        raise ValueError('@modifier must be called with only keyword arguments, @modifier(name=\'foo\'), or no arguments, @modifier; ({0:d} given)'.format(len(args)))

from .histosys import histosys
from .normfactor import normfactor
from .normsys import normsys
from .shapefactor import shapefactor
from .shapesys import shapesys
from .staterror import staterror
__all__ = ['histosys','normfactor','normsys','shapefactor','shapesys','staterror']
