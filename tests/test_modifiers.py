import pytest
import sys
import inspect
from six import with_metaclass

import pyhf

modifiers_to_test = ["histosys", "normfactor", "normsys", "shapefactor", "shapesys"]

# we make sure we can import all of our pre-defined modifiers correctly
@pytest.mark.parametrize("test_modifier", modifiers_to_test)
def test_import_default_modifiers(test_modifier):
    modifier = getattr(__import__('pyhf.modifiers', fromlist=[test_modifier]), test_modifier)
    assert test_modifier in pyhf.modifiers.registry
    assert callable(pyhf.modifiers.registry[test_modifier])

# we make sure decorate can use auto-naming
def test_decorate_modifier_name_auto():
    from pyhf.modifiers import modifier

    @modifier
    class myCustomModifier(object):
        pass

    assert inspect.isclass(myCustomModifier)
    assert 'myCustomModifier' in pyhf.modifiers.registry
    assert pyhf.modifiers.registry['myCustomModifier'] == myCustomModifier
    del pyhf.modifiers.registry['myCustomModifier']

# we make sure decorate allows for custom naming
def test_decorate_modifier_name_custom():
    from pyhf.modifiers import modifier

    @modifier('myCustomName')
    class myCustomModifier(object):
        pass

    assert inspect.isclass(myCustomModifier)
    assert 'myCustomModifier' not in pyhf.modifiers.registry
    assert 'myCustomName' in pyhf.modifiers.registry
    assert pyhf.modifiers.registry['myCustomName'] == myCustomModifier
    del pyhf.modifiers.registry['myCustomName']

# we make sure decorate raises errors if passed more than one argument, or not a string
def test_decorate_wrong_values():
    from pyhf.modifiers import modifier

    with pytest.raises(ValueError):
        @modifier('too','many','args')
        class myCustomModifier(object):
            pass

    with pytest.raises(TypeError):
        @modifier(1.5)
        class myCustomModifier(object):
            pass

# we catch name clashes when adding duplicate names for modifiers
def test_registry_name_clash():
    from pyhf.modifiers import modifier

    with pytest.raises(KeyError):
        @modifier('histosys')
        class myCustomModifier(object):
            pass

    with pytest.raises(KeyError):
        class myCustomModifier(object):
            pass

        pyhf.modifiers.add_to_registry(myCustomModifier, 'histosys')
