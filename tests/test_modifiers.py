import pytest
import sys
import inspect
from six import with_metaclass

import pyhf

modifiers_to_test = ["histosys", "normfactor", "normsys", "shapefactor", "shapesys"]

'''
This is a teardown fixture to try and remove all traces of pyhf.<foo> from
python's module-cache.

It effectively clean the slate for a new test session so that the modifier
registry is empty after tests and constraints that are re-imported are
re-added.
'''
@pytest.fixture
def importer():
    def empty_registry():
      from pyhf import modifiers
      for k in list(modifiers.registry.keys()):
          del modifiers.registry[k]
      for k in list(sys.modules.keys()):
          if k.startswith('pyhf'): del sys.modules[k]
    empty_registry()
    yield "importer"
    empty_registry()

# we make sure the registry is empty by default
def test_empty_registry(importer):
    from pyhf import modifiers
    assert modifiers.registry == {}

# we make sure we can import all of our pre-defined modifiers correctly
@pytest.mark.parametrize("test_modifier", modifiers_to_test)
def test_import_default_modifiers(importer, test_modifier):
    from pyhf import modifiers
    modifier = getattr(__import__('pyhf.modifiers', fromlist=[test_modifier]), test_modifier)
    assert test_modifier in modifiers.registry
    assert callable(modifiers.registry[test_modifier])

# we make sure decorate can use auto-naming
def test_decorate_modifier_name_auto(importer):
    from pyhf import modifiers
    from pyhf.modifiers import modifier

    @modifier
    class myCustomModifier(object):
        pass

    assert inspect.isclass(myCustomModifier)
    assert 'myCustomModifier' in modifiers.registry
    assert modifiers.registry['myCustomModifier'] == myCustomModifier

# we make sure decorate allows for custom naming
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

# we make sure decorate raises errors if passed more than one argument, or not a string
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

# we catch name clashes when adding duplicate names for modifiers
def test_registry_name_clash(importer):
    from pyhf import modifiers
    from pyhf.modifiers import modifier
    from pyhf.modifiers import histosys

    assert 'histosys' in modifiers.registry

    with pytest.raises(KeyError):
        @modifier('histosys')
        class myCustomModifier(object):
            pass

    with pytest.raises(KeyError):
        class myCustomModifier(object):
            pass

        modifiers.add_to_registry(myCustomModifier, 'histosys')
