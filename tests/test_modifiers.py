import pytest
import sys
import inspect
from six import with_metaclass

import pyhf

modifiers_to_test = ["histosys", "normsys", "shapesys"]

@pytest.fixture
def importer():
    yield "importer"
    from pyhf import modifiers
    for k in list(modifiers.registry.keys()):
        del modifiers.registry[k]
    for k in list(sys.modules.keys()):
        if k.startswith('pyhf'): del sys.modules[k]

def test_empty_registry(importer):
    from pyhf import modifiers
    assert modifiers.registry == {}

@pytest.mark.parametrize("test_modifier", modifiers_to_test)
def test_import_histosys(importer, test_modifier):
    from pyhf import modifiers
    modifier = getattr(__import__('pyhf.modifiers', fromlist=[test_modifier]), test_modifier)
    assert '{0:s}_constraint'.format(test_modifier) in modifiers.registry
    assert isinstance(modifiers.registry['{0:s}_constraint'.format(test_modifier)], modifiers.IModifier)

def test_decorate_modifier_name_auto(importer):
    from pyhf import modifiers
    from pyhf.modifiers import modifier

    @modifier
    class myCustomModifier(object):
        pass

    assert inspect.isclass(myCustomModifier)
    assert 'myCustomModifier' in modifiers.registry
    assert modifiers.registry['myCustomModifier'] == myCustomModifier

def test_decorate_modifier_name_custom(importer):
    from pyhf import modifiers
    from pyhf.modifiers import modifier

    @modifier('myCustomName')
    class myCustomModifier(object):
        pass

    assert inspect.isclass(myCustomModifier)
    assert 'myCustomModifier' not in modifiers.registry
    assert 'myCustomName' in modifiers.registry
    assert modifiers.registry['myCustomName'] == myCustomModifier

def test_decorate_wrong_values(importer):
    from pyhf import modifiers
    from pyhf.modifiers import modifier

    with pytest.raises(ValueError):
        @modifier('too','many','args')
        class myCustomModifier(object):
            pass

    with pytest.raises(TypeError):
        @modifier(1.5)
        class myCustomModifier(object):
            pass

def test_modifier_metaclass(importer):
    from pyhf import modifiers

    class myCustomModifier(with_metaclass(modifiers.IModifier, object)):
        pass

    assert inspect.isclass(myCustomModifier)
    assert 'myCustomModifier' in modifiers.registry
    assert modifiers.registry['myCustomModifier'] == myCustomModifier

def test_registry_name_clash(importer):
    from pyhf import modifiers
    from pyhf.modifiers import modifier
    from pyhf.modifiers import histosys

    assert 'histosys_constraint' in modifiers.registry

    with pytest.raises(KeyError):
        @modifier('histosys_constraint')
        class myCustomModifier(object):
            pass

    with pytest.raises(KeyError):
        class histosys_constraint(with_metaclass(modifiers.IModifier, object)):
            pass
