import pyhf
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


def test_torootname():
    m1 = pyhf.simplemodels.correlated_background([5],[50],[52],[48])
    m2 = pyhf.simplemodels.hepdata_like([5],[50],[7])
    m3 = pyhf.simplemodels.hepdata_like([5,6],[50,50],[7,8])

    assert pyhf.compat.parset_to_rootnames(
        m1.config.param_set('mu')
    ) == 'mu'

    assert pyhf.compat.parset_to_rootnames(
        m1.config.param_set('correlated_bkg_uncertainty')
    ) == 'alpha_correlated_bkg_uncertainty'

    assert pyhf.compat.parset_to_rootnames(
        m2.config.param_set('uncorr_bkguncrt')
    ) == ['gamma_uncorr_bkguncrt_0']

    assert pyhf.compat.parset_to_rootnames(
        m3.config.param_set('uncorr_bkguncrt')
    ) == ['gamma_uncorr_bkguncrt_0','gamma_uncorr_bkguncrt_1']