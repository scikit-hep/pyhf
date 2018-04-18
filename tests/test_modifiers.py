import pytest
import sys
import inspect
from six import with_metaclass

import pyhf

modifiers_to_test = ["histosys", "normfactor", "normsys", "shapefactor", "shapesys"]

# we make sure we can import all of our pre-defined modifiers correctly
@pytest.mark.parametrize("test_modifier", modifiers_to_test)
def test_import_default_modifiers(test_modifier):
    modifier = pyhf.modifiers.registry.get(test_modifier, None)
    assert test_modifier in pyhf.modifiers.registry
    assert modifier is not None
    assert callable(modifier)
    assert hasattr(modifier, 'is_constrained')


# we make sure modifiers have right structure
def test_modifiers_structure():
    from pyhf.modifiers import modifier

    @modifier(name='myUnconstrainedModifier')
    class myCustomModifier(object):
        def __init__(self): pass
        def add_sample(self): pass

    assert inspect.isclass(myCustomModifier)
    assert 'myUnconstrainedModifier' in pyhf.modifiers.registry
    assert pyhf.modifiers.registry['myUnconstrainedModifier'] == myCustomModifier
    assert pyhf.modifiers.registry['myUnconstrainedModifier'].is_constrained == False
    assert pyhf.modifiers.registry['myUnconstrainedModifier'].is_shared == False
    del pyhf.modifiers.registry['myUnconstrainedModifier']

    @modifier(name='mySharedModifier', shared=True)
    class myCustomModifier(object):
        def __init__(self): pass
        def add_sample(self): pass

    assert inspect.isclass(myCustomModifier)
    assert 'mySharedModifier' in pyhf.modifiers.registry
    assert pyhf.modifiers.registry['mySharedModifier'] == myCustomModifier
    assert pyhf.modifiers.registry['mySharedModifier'].is_shared == True
    del pyhf.modifiers.registry['mySharedModifier']

    @modifier(name='myConstrainedModifier', constrained=True)
    class myCustomModifier(object):
        def __init__(self): pass
        def add_sample(self): pass
        def pdf(self): pass
        def alphas(self): pass
        def expected_data(self): pass

    assert inspect.isclass(myCustomModifier)
    assert 'myConstrainedModifier' in pyhf.modifiers.registry
    assert pyhf.modifiers.registry['myConstrainedModifier'] == myCustomModifier
    assert pyhf.modifiers.registry['myConstrainedModifier'].is_constrained == True
    assert pyhf.modifiers.registry['myConstrainedModifier'].is_shared == False
    del pyhf.modifiers.registry['myConstrainedModifier']

    with pytest.raises(pyhf.exceptions.InvalidModifier):
        @modifier
        class myCustomModifier(object):
            pass

    with pytest.raises(pyhf.exceptions.InvalidModifier):
        @modifier(constrained=True)
        class myCustomModifier(object):
            pass

    with pytest.raises(pyhf.exceptions.InvalidModifier):
        @modifier(name='myConstrainedModifier', constrained=True)
        class myCustomModifier(object):
            def __init__(self): pass
            def add_sample(self): pass


# we make sure decorate can use auto-naming
def test_modifier_name_auto():
    from pyhf.modifiers import modifier

    @modifier
    class myCustomModifier(object):
        def __init__(self): pass
        def add_sample(self): pass

    assert inspect.isclass(myCustomModifier)
    assert 'myCustomModifier' in pyhf.modifiers.registry
    assert pyhf.modifiers.registry['myCustomModifier'] == myCustomModifier
    del pyhf.modifiers.registry['myCustomModifier']


# we make sure decorate can use auto-naming with keyword arguments
def test_modifier_name_auto_withkwargs():
    from pyhf.modifiers import modifier

    @modifier(name=None, constrained=False)
    class myCustomModifier(object):
        def __init__(self): pass
        def add_sample(self): pass

    assert inspect.isclass(myCustomModifier)
    assert 'myCustomModifier' in pyhf.modifiers.registry
    assert pyhf.modifiers.registry['myCustomModifier'] == myCustomModifier
    del pyhf.modifiers.registry['myCustomModifier']


# we make sure decorate allows for custom naming
def test_modifier_name_custom():
    from pyhf.modifiers import modifier

    @modifier(name='myCustomName')
    class myCustomModifier(object):
        def __init__(self):
            pass

        def add_sample(self):
            pass

    assert inspect.isclass(myCustomModifier)
    assert 'myCustomModifier' not in pyhf.modifiers.registry
    assert 'myCustomName' in pyhf.modifiers.registry
    assert pyhf.modifiers.registry['myCustomName'] == myCustomModifier
    del pyhf.modifiers.registry['myCustomName']


# we make sure decorate raises errors if passed more than one argument, or not a string
def test_decorate_with_wrong_values():
    from pyhf.modifiers import modifier

    with pytest.raises(ValueError):
        @modifier('too','many','args')
        class myCustomModifier(object):
            pass

    with pytest.raises(TypeError):
        @modifier(name=1.5)
        class myCustomModifier(object):
            pass

    with pytest.raises(ValueError):
        @modifier(unused='arg')
        class myCustomModifier(object):
            pass


# we catch name clashes when adding duplicate names for modifiers
def test_registry_name_clash():
    from pyhf.modifiers import modifier

    with pytest.raises(KeyError):
        @modifier(name='histosys')
        class myCustomModifier(object):
            pass

    with pytest.raises(KeyError):
        class myCustomModifier(object):
            pass

        pyhf.modifiers.add_to_registry(myCustomModifier, 'histosys')
