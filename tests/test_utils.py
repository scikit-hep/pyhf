import os
import pytest

import pyhf
import pyhf.simplemodels
import pyhf.utils


def test_get_default_schema():
    assert os.path.isfile(pyhf.utils.get_default_schema())


def test_load_default_schema():
    assert pyhf.utils.load_schema(pyhf.utils.get_default_schema())


def test_load_missing_schema():
    with pytest.raises(IOError):
        pyhf.utils.load_schema('a/fake/path/that/should/not/work.json')


def test_load_custom_schema(tmpdir):
    temp = tmpdir.join("custom_schema.json")
    temp.write('{"foo": "bar"}')
    assert pyhf.utils.load_schema(temp.strpath)


def test_hypotest(tmpdir):
    """
    Check that the return structure of pyhf.utils.hypotest is as expected
    """
    tb = pyhf.tensorlib

    def check_uniform_type(in_list):
        return all([isinstance(item, type(tb.astensor(item))) for item in in_list])

    pdf = pyhf.simplemodels.hepdata_like(
        signal_data=[12.0, 11.0], bkg_data=[50.0, 52.0], bkg_uncerts=[3.0, 7.0]
    )
    mu_test = 1.0
    data = [51, 48] + pdf.config.auxdata
    kwargs = {}

    result = pyhf.utils.hypotest(mu_test, data, pdf, **kwargs)
    # CLs_obs
    assert len(list(result)) == 1
    assert isinstance(result, type(tb.astensor(result)))

    kwargs['return_p_values'] = True
    result = pyhf.utils.hypotest(mu_test, data, pdf, **kwargs)
    # CLs_obs, [CL_sb, CL_b]
    assert len(list(result)) == 2
    assert isinstance(result[0], type(tb.astensor(result[0])))
    assert len(result[1]) == 2
    assert check_uniform_type(result[1])

    kwargs['return_expected'] = True
    result = pyhf.utils.hypotest(mu_test, data, pdf, **kwargs)
    # CLs_obs, [CLsb, CLb], CLs_exp
    assert len(list(result)) == 3
    assert isinstance(result[0], type(tb.astensor(result[0])))
    assert len(result[1]) == 2
    assert check_uniform_type(result[1])
    assert isinstance(result[2], type(tb.astensor(result[2])))

    kwargs['return_expected_set'] = True
    result = pyhf.utils.hypotest(mu_test, data, pdf, **kwargs)
    # CLs_obs, [CLsb, CLb], CLs_exp, CLs_exp @[-2, -1, 0, +1, +2]sigma
    assert len(list(result)) == 4
    assert isinstance(result[0], type(tb.astensor(result[0])))
    assert len(result[1]) == 2
    assert check_uniform_type(result[1])
    assert isinstance(result[2], type(tb.astensor(result[2])))
    assert len(result[3]) == 5
    assert check_uniform_type(result[3])

    kwargs['return_test_statistics'] = True
    result = pyhf.utils.hypotest(mu_test, data, pdf, **kwargs)
    # CLs_obs, [CLsb, CLb], CLs_exp, CLs_exp @[-2, -1, 0, +1, +2]sigma, [q_mu, q_mu_Asimov]
    assert len(list(result)) == 5
    assert isinstance(result[0], type(tb.astensor(result[0])))
    assert len(result[1]) == 2
    assert check_uniform_type(result[1])
    assert isinstance(result[2], type(tb.astensor(result[2])))
    assert len(result[3]) == 5
    assert check_uniform_type(result[3])
    assert len(result[4]) == 2
    assert check_uniform_type(result[4])
