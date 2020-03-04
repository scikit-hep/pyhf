import pytest
import pyhf
import numpy as np


@pytest.fixture(scope='module')
def hypotest_args():
    pdf = pyhf.simplemodels.hepdata_like(
        signal_data=[12.0, 11.0], bkg_data=[50.0, 52.0], bkg_uncerts=[3.0, 7.0]
    )
    mu_test = 1.0
    data = [51, 48] + pdf.config.auxdata
    return mu_test, data, pdf


def check_uniform_type(in_list):
    return all(
        [isinstance(item, type(pyhf.tensorlib.astensor(item))) for item in in_list]
    )


def test_hypotest_default(tmpdir, hypotest_args):
    """
    Check that the default return structure of pyhf.infer.hypotest is as expected
    """
    tb = pyhf.tensorlib

    kwargs = {}
    result = pyhf.infer.hypotest(*hypotest_args, **kwargs)
    # CLs_obs
    assert len(list(result)) == 1
    assert isinstance(result, type(tb.astensor(result)))


def test_hypotest_return_tail_probs(tmpdir, hypotest_args):
    """
    Check that the return structure of pyhf.infer.hypotest with the
    return_tail_probs keyword arg is as expected
    """
    tb = pyhf.tensorlib

    kwargs = {'return_tail_probs': True}
    result = pyhf.infer.hypotest(*hypotest_args, **kwargs)
    # CLs_obs, [CL_sb, CL_b]
    assert len(list(result)) == 2
    assert isinstance(result[0], type(tb.astensor(result[0])))
    assert len(result[1]) == 2
    assert check_uniform_type(result[1])


def test_hypotest_return_expected(tmpdir, hypotest_args):
    """
    Check that the return structure of pyhf.infer.hypotest with the
    additon of the return_expected keyword arg is as expected
    """
    tb = pyhf.tensorlib

    kwargs = {'return_tail_probs': True, 'return_expected': True}
    result = pyhf.infer.hypotest(*hypotest_args, **kwargs)
    # CLs_obs, [CLsb, CLb], CLs_exp
    assert len(list(result)) == 3
    assert isinstance(result[0], type(tb.astensor(result[0])))
    assert len(result[1]) == 2
    assert check_uniform_type(result[1])
    assert isinstance(result[2], type(tb.astensor(result[2])))


def test_hypotest_return_expected_set(tmpdir, hypotest_args):
    """
    Check that the return structure of pyhf.infer.hypotest with the
    additon of the return_expected_set keyword arg is as expected
    """
    tb = pyhf.tensorlib

    kwargs = {
        'return_tail_probs': True,
        'return_expected': True,
        'return_expected_set': True,
    }
    result = pyhf.infer.hypotest(*hypotest_args, **kwargs)
    # CLs_obs, [CLsb, CLb], CLs_exp, CLs_exp @[-2, -1, 0, +1, +2]sigma
    assert len(list(result)) == 4
    assert isinstance(result[0], type(tb.astensor(result[0])))
    assert len(result[1]) == 2
    assert check_uniform_type(result[1])
    assert isinstance(result[2], type(tb.astensor(result[2])))
    assert len(result[3]) == 5
    assert check_uniform_type(result[3])


def test_inferapi_pyhf_independence():
    '''
    pyhf.infer should eventually be factored out so it should be
    infependent from pyhf internals. This is testing that
    a much simpler model still can run through pyhf.infer.hypotest
    '''
    from pyhf import get_backend

    class _NonPyhfConfig(object):
        def __init__(self):
            self.poi_index = 0
            self.npars = 2

        def suggested_init(self):
            return [1.0, 1.0]

        def suggested_bounds(self):
            return [[0.0, 10.0], [0.0, 10.0]]

    class NonPyhfModel(object):
        def __init__(self, spec):
            self.sig, self.nominal, self.uncert = spec
            self.factor = (self.nominal / self.uncert) ** 2
            self.aux = 1.0 * self.factor
            self.config = _NonPyhfConfig()

        def _make_main_pdf(self, pars):
            mu, gamma = pars
            expected_main = gamma * self.nominal + mu * self.sig
            return pyhf.probability.Poisson(expected_main)

        def _make_constraint_pdf(self, pars):
            mu, gamma = pars
            return pyhf.probability.Poisson(gamma * self.factor)

        def expected_data(self, pars, include_auxdata=True):
            tensorlib, _ = get_backend()
            expected_main = tensorlib.astensor(
                [self._make_main_pdf(pars).expected_data()]
            )
            aux_data = tensorlib.astensor(
                [self._make_constraint_pdf(pars).expected_data()]
            )
            if not include_auxdata:
                return expected_main
            return tensorlib.concatenate([expected_main, aux_data])

        def logpdf(self, pars, data):
            tensorlib, _ = get_backend()
            maindata, auxdata = data
            main = self._make_main_pdf(pars).log_prob(maindata)
            constraint = self._make_constraint_pdf(pars).log_prob(auxdata)
            return tensorlib.astensor([main + constraint])

    model = NonPyhfModel([5, 50, 7])
    cls = pyhf.infer.hypotest(
        1.0, model.expected_data(model.config.suggested_init()), model
    )

    assert np.isclose(cls[0], 0.7267836451638846)
