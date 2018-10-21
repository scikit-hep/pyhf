from six import string_types
import logging
log = logging.getLogger(__name__)

from .. import exceptions
from .. import get_backend

registry = {}

'''
Check if given object contains the right structure for modifiers
'''
def validate_modifier_structure(modifier):
    required_methods = ['required_parset']

    for method in required_methods:
        if not hasattr(modifier, method):
            raise exceptions.InvalidModifier('Expected {0:s} method on modifier {1:s}'.format(method, modifier.__name__))
    return True

'''
Consistent add_to_registry() function that handles actually adding thing to the registry.

Raises an error if the name to register for the modifier already exists in the registry,
or if the modifier does not have the right structure.
'''
def add_to_registry(cls, cls_name=None, constrained=False, pdf_type='normal', op_code='addition'):
    global registry
    cls_name = cls_name or cls.__name__
    if cls_name in registry: raise KeyError('The modifier name "{0:s}" is already taken.'.format(cls_name))
    # validate the structure
    validate_modifier_structure(cls)
    # set is_constrained
    cls.is_constrained = constrained
    if constrained:
        tensorlib, _ = get_backend()
        if not hasattr(tensorlib, pdf_type):
            raise exceptions.InvalidModifier('The specified pdf_type "{0:s}" is not valid for {1:s}({2:s}). See pyhf.tensor documentation for available pdfs.'.format(pdf_type, cls_name, cls.__name__))
        cls.pdf_type = pdf_type
    else:
        cls.pdf_type = None

    if op_code not in ['addition', 'multiplication']:
        raise exceptions.InvalidModifier('The specified op_code "{0:s}" is not valid for {1:s}({2:s}). See pyhf.modifier documentation for available operation codes.'.format(op_code, cls_name, cls.__name__))
    cls.op_code = op_code

    registry[cls_name] = cls

'''
Decorator for registering modifiers. To flag the modifier as a constrained modifier, add `constrained=True`.


Args:
    name: the name of the modifier to use. Use the class name by default. (default: None)
    constrained: whether the modifier is constrained or not. (default: False)
    pdf_type: the name of the pdf to use from tensorlib if constrained. (default: normal)
    op_code: the name of the operation the modifier performs on the data (e.g. addition, multiplication)

Returns:
    modifier

Raises:
    ValueError: too many keyword arguments, or too many arguments, or wrong arguments
    TypeError: provided name is not a string
    pyhf.exceptions.InvalidModifier: object does not have necessary modifier structure

Examples:

  >>> @modifiers.modifier
  >>> ... class myCustomModifier(object):
  >>> ...   @classmethod
  >>> ...   def required_parset(cls, npars): pass

  >>> @modifiers.modifier(name='myCustomNamer')
  >>> ... class myCustomModifier(object):
  >>> ...   @classmethod
  >>> ...   def required_parset(cls, npars): pass

  >>> @modifiers.modifier(constrained=False)
  >>> ... class myUnconstrainedModifier(object):
  >>> ...   @classmethod
  >>> ...   def required_parset(cls, npars): pass
  >>> ...
  >>> myUnconstrainedModifier.pdf_type
  None

  >>> @modifiers.modifier(constrained=True, pdf_type='poisson')
  >>> ... class myConstrainedCustomPoissonModifier(object):
  >>> ...   @classmethod
  >>> ...   def required_parset(cls, npars): pass
  >>> ...
  >>> myConstrainedCustomGaussianModifier.pdf_type
  'poisson'

  >>> @modifiers.modifier(constrained=True)
  >>> ... class myCustomModifier(object):
  >>> ...   @classmethod
  >>> ...   def required_parset(cls, npars): pass

  >>> @modifiers.modifier(op_code='multiplication')
  >>> ... class myMultiplierModifier(object):
  >>> ...   @classmethod
  >>> ...   def required_parset(cls, npars): pass
  >>> ...
  >>> myMultiplierModifier.op_code
  'multiplication'

'''
def modifier(*args, **kwargs):
    name = kwargs.pop('name', None)
    constrained = bool(kwargs.pop('constrained', False))
    pdf_type = str(kwargs.pop('pdf_type', 'normal'))
    op_code = str(kwargs.pop('op_code', 'addition'))
    # check for unparsed keyword arguments
    if len(kwargs) != 0:
        raise ValueError('Unparsed keyword arguments {}'.format(kwargs.keys()))
    # check to make sure the given name is a string, if passed in one
    if not isinstance(name, string_types) and name is not None:
        raise TypeError('@modifier must be given a string. You gave it {}'.format(type(name)))

    def _modifier(name, constrained, pdf_type, op_code):
        def wrapper(cls):
            add_to_registry(cls, cls_name=name, constrained=constrained, pdf_type=pdf_type, op_code=op_code)
            return cls
        return wrapper

    if len(args) == 0:
        # called like @modifier(name='foo', constrained=False, pdf_type='normal', op_code='addition')
        return _modifier(name, constrained, pdf_type, op_code)
    elif len(args) == 1:
        # called like @modifier
        if not callable(args[0]):
            raise ValueError('You must decorate a callable python object')
        add_to_registry(args[0], cls_name=name, constrained=constrained, pdf_type=pdf_type, op_code=op_code)
        return args[0]
    else:
        raise ValueError('@modifier must be called with only keyword arguments, @modifier(name=\'foo\'), or no arguments, @modifier; ({0:d} given)'.format(len(args)))

from .histosys import histosys, histosys_combined
from .normfactor import normfactor, normfactor_combined
from .normsys import normsys, normsys_combined
from .shapefactor import shapefactor, shapefactor_combined
from .shapesys import shapesys, shapesys_combined
from .staterror import staterror, staterror_combined
combined = {
    'normsys': normsys_combined,
    'histosys': histosys_combined,
    'normfactor': normfactor_combined,
    'staterror': staterror_combined,
    'shapefactor': shapefactor_combined,
    'shapesys': shapesys_combined
}
__all__ = [
    'histosys','histosys_combined',
    'normfactor','normfactor_combined',
    'normsys','normsys_combined',
    'shapefactor','shapefactor_combined',
    'shapesys','shapesys_combined',
    'staterror','staterror_combined',
    'combined'
]
