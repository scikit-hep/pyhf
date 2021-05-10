import pytest
import pyhf
import pyhf.compat
import pyhf.readxml


def test_interpretation():
    interp = pyhf.compat.interpret_rootname('gamma_foo_0')
    assert interp['constrained'] == 'n/a'
    assert not interp['is_scalar']
    assert interp['name'] == 'foo'
    assert interp['element'] == 0

    interp = pyhf.compat.interpret_rootname('alpha_foo')
    assert interp['constrained']
    assert interp['is_scalar']
    assert interp['name'] == 'foo'
    assert interp['element'] == 'n/a'

    interp = pyhf.compat.interpret_rootname('mu')
    assert not interp['constrained']
    assert interp['is_scalar']
    assert interp['name'] == 'mu'
    assert interp['element'] == 'n/a'

    interp = pyhf.compat.interpret_rootname('Lumi')
    assert interp['name'] == 'lumi'

    interp = pyhf.compat.interpret_rootname('Lumi')
    assert interp['name'] == 'lumi'

    with pytest.raises(ValueError):
        pyhf.compat.interpret_rootname('gamma_foo')

    with pytest.raises(ValueError):
        pyhf.compat.interpret_rootname('alpha_')


def test_torootname():
    m1 = pyhf.simplemodels.correlated_background([5], [50], [52], [48])
    m2 = pyhf.simplemodels.uncorrelated_background([5], [50], [7])
    m3 = pyhf.simplemodels.uncorrelated_background([5, 6], [50, 50], [7, 8])

    assert pyhf.compat.paramset_to_rootnames(m1.config.param_set('mu')) == 'mu'

    assert (
        pyhf.compat.paramset_to_rootnames(
            m1.config.param_set('correlated_bkg_uncertainty')
        )
        == 'alpha_correlated_bkg_uncertainty'
    )

    assert pyhf.compat.paramset_to_rootnames(
        m2.config.param_set('uncorr_bkguncrt')
    ) == ['gamma_uncorr_bkguncrt_0']

    assert pyhf.compat.paramset_to_rootnames(
        m3.config.param_set('uncorr_bkguncrt')
    ) == [
        'gamma_uncorr_bkguncrt_0',
        'gamma_uncorr_bkguncrt_1',
    ]


def test_fromxml():
    parsed_xml = pyhf.readxml.parse(
        'validation/xmlimport_input3/config/examples/example_ShapeSys.xml',
        'validation/xmlimport_input3',
    )

    # build the spec, strictly checks properties included
    spec = {
        'channels': parsed_xml['channels'],
        'parameters': parsed_xml['measurements'][0]['config']['parameters'],
    }
    m = pyhf.Model(spec, poi_name='SigXsecOverSM')

    assert pyhf.compat.paramset_to_rootnames(m.config.param_set('lumi')) == 'Lumi'
