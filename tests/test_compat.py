import pyhf.compat

def test_interpretation():
    interp = pyhf.compat.interpret_rootname('gamma_foo_0')
    assert interp['constrained'] == 'n/a'
    assert interp['is_scalar'] == False
    assert interp['name'] == 'foo'
    assert interp['element'] == 0
    
    interp = pyhf.compat.interpret_rootname('alpha_foo')
    assert interp['constrained'] == True
    assert interp['is_scalar'] == True
    assert interp['name'] == 'foo'
    assert interp['element'] == 'n/a'

    interp = pyhf.compat.interpret_rootname('mu')
    assert interp['constrained'] == False
    assert interp['is_scalar'] == True
    assert interp['name'] == 'mu'
    assert interp['element'] == 'n/a'
