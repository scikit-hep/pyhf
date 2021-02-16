import logging

from .. import exceptions
from .. import get_backend

log = logging.getLogger(__name__)

registry = {}


def validate_modifier_structure(modifier):
    """
    Check if given object contains the right structure for modifiers
    """
    required_methods = ['required_parset']

    for method in required_methods:
        if not hasattr(modifier, method):
            raise exceptions.InvalidModifier(
                f'Expected {method:s} method on modifier {modifier.__name__:s}'
            )
    return True


def add_to_registry(
    cls, cls_name=None, constrained=False, pdf_type='normal', op_code='addition'
):
    """
    Consistent add_to_registry() function that handles actually adding thing to the registry.

    Raises an error if the name to register for the modifier already exists in the registry,
    or if the modifier does not have the right structure.
    """
    global registry
    cls_name = cls_name or cls.__name__
    if cls_name in registry:
        raise KeyError(f'The modifier name "{cls_name:s}" is already taken.')
    # validate the structure
    validate_modifier_structure(cls)
    # set is_constrained
    cls.is_constrained = constrained
    if constrained:
        tensorlib, _ = get_backend()
        if not hasattr(tensorlib, pdf_type):
            raise exceptions.InvalidModifier(
                f'The specified pdf_type "{pdf_type:s}" is not valid for {cls_name:s}({cls.__name__:s}). See pyhf.tensor documentation for available pdfs.'
            )
        cls.pdf_type = pdf_type
    else:
        cls.pdf_type = None

    if op_code not in ['addition', 'multiplication']:
        raise exceptions.InvalidModifier(
            f'The specified op_code "{op_code:s}" is not valid for {cls_name:s}({cls.__name__:s}). See pyhf.modifier documentation for available operation codes.'
        )
    cls.op_code = op_code

    registry[cls_name] = cls


def modifier(*args, **kwargs):
    """
    Decorator for registering modifiers. To flag the modifier as a constrained modifier, add `constrained=True`.


    Args:
        name (:obj:`str`): the name of the modifier to use. Use the class name by default. (default: None)
        constrained (:obj:`bool`): whether the modifier is constrained or not. (default: False)
        pdf_type (:obj:`str): the name of the pdf to use from tensorlib if constrained. (default: normal)
        op_code (:obj:`str`): the name of the operation the modifier performs on the data (e.g. addition, multiplication)

    Returns:
        modifier

    Raises:
        ValueError: too many keyword arguments, or too many arguments, or wrong arguments
        TypeError: provided name is not a string
        pyhf.exceptions.InvalidModifier: object does not have necessary modifier structure
    """
    #
    # Examples:
    #
    #   >>> @modifiers.modifier
    #   >>> ... class myCustomModifier(object):
    #   >>> ...   @classmethod
    #   >>> ...   def required_parset(cls, sample_data, modifier_data): pass
    #
    #   >>> @modifiers.modifier(name='myCustomNamer')
    #   >>> ... class myCustomModifier(object):
    #   >>> ...   @classmethod
    #   >>> ...   def required_parset(cls, sample_data, modifier_data): pass
    #
    #   >>> @modifiers.modifier(constrained=False)
    #   >>> ... class myUnconstrainedModifier(object):
    #   >>> ...   @classmethod
    #   >>> ...   def required_parset(cls, sample_data, modifier_data): pass
    #   >>> ...
    #   >>> myUnconstrainedModifier.pdf_type
    #   None
    #
    #   >>> @modifiers.modifier(constrained=True, pdf_type='poisson')
    #   >>> ... class myConstrainedCustomPoissonModifier(object):
    #   >>> ...   @classmethod
    #   >>> ...   def required_parset(cls, sample_data, modifier_data): pass
    #   >>> ...
    #   >>> myConstrainedCustomGaussianModifier.pdf_type
    #   'poisson'
    #
    #   >>> @modifiers.modifier(constrained=True)
    #   >>> ... class myCustomModifier(object):
    #   >>> ...   @classmethod
    #   >>> ...   def required_parset(cls, sample_data, modifier_data): pass
    #
    #   >>> @modifiers.modifier(op_code='multiplication')
    #   >>> ... class myMultiplierModifier(object):
    #   >>> ...   @classmethod
    #   >>> ...   def required_parset(cls, sample_data, modifier_data): pass
    #   >>> ...
    #   >>> myMultiplierModifier.op_code
    #   'multiplication'

    def _modifier(name, constrained, pdf_type, op_code):
        def wrapper(cls):
            add_to_registry(
                cls,
                cls_name=name,
                constrained=constrained,
                pdf_type=pdf_type,
                op_code=op_code,
            )
            return cls

        return wrapper

    name = kwargs.pop('name', None)
    constrained = bool(kwargs.pop('constrained', False))
    pdf_type = str(kwargs.pop('pdf_type', 'normal'))
    op_code = str(kwargs.pop('op_code', 'addition'))
    # check for unparsed keyword arguments
    if kwargs:
        raise ValueError(f'Unparsed keyword arguments {kwargs.keys()}')
    # check to make sure the given name is a string, if passed in one
    if not isinstance(name, str) and name is not None:
        raise TypeError(f'@modifier must be given a string. You gave it {type(name)}')

    if not args:
        # called like @modifier(name='foo', constrained=False, pdf_type='normal', op_code='addition')
        return _modifier(name, constrained, pdf_type, op_code)
    if len(args) == 1:
        # called like @modifier
        if not callable(args[0]):
            raise ValueError('You must decorate a callable python object')
        add_to_registry(
            args[0],
            cls_name=name,
            constrained=constrained,
            pdf_type=pdf_type,
            op_code=op_code,
        )
        return args[0]
    raise ValueError(
        f'@modifier must be called with only keyword arguments, @modifier(name=\'foo\'), or no arguments, @modifier; ({len(args):d} given)'
    )


from .histosys import histosys, histosys_combined
from .lumi import lumi, lumi_combined
from .normfactor import normfactor, normfactor_combined
from .normsys import normsys, normsys_combined
from .shapefactor import shapefactor, shapefactor_combined
from .shapesys import shapesys, shapesys_combined
from .staterror import staterror, staterror_combined

uncombined = {
    'histosys': histosys,
    'lumi': lumi,
    'normfactor': normfactor,
    'normsys': normsys,
    'shapefactor': shapefactor,
    'shapesys': shapesys,
    'staterror': staterror,
}

combined = {
    'histosys': histosys_combined,
    'lumi': lumi_combined,
    'normfactor': normfactor_combined,
    'normsys': normsys_combined,
    'shapefactor': shapefactor_combined,
    'shapesys': shapesys_combined,
    'staterror': staterror_combined,
}

__all__ = [
    'histosys',
    'histosys_combined',
    'lumi',
    'lumi_combined',
    'normfactor',
    'normfactor_combined',
    'normsys',
    'normsys_combined',
    'shapefactor',
    'shapefactor_combined',
    'shapesys',
    'shapesys_combined',
    'staterror',
    'staterror_combined',
    'combined',
]
