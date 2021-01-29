import pytest
import pyhf
import numpy as np
import scipy.stats


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


def test_upperlimit(tmpdir, hypotest_args):
    """
    Check that the default return structure of pyhf.infer.hypotest is as expected
    """
    _, data, model = hypotest_args
    results = pyhf.infer.intervals.upperlimit(data, model, scan=np.linspace(0, 5, 11))
    assert len(results) == 2
    observed_limit, expected_limits = results
    assert observed_limit == pytest.approx(1.0262717528250018, rel=1e-5)
    assert expected_limits == pytest.approx(
        [0.65765824, 0.87999803, 1.12454316, 1.50244107, 2.09233767], rel=1e-5
    )


def test_mle_fit_default(tmpdir, hypotest_args):
    """
    Check that the default return structure of pyhf.infer.mle.fit is as expected
    """
    tb = pyhf.tensorlib

    _, data, model = hypotest_args
    kwargs = {}
    result = pyhf.infer.mle.fit(data, model, **kwargs)
    # bestfit_pars
    assert isinstance(result, type(tb.astensor(result)))
    assert pyhf.tensorlib.shape(result) == (model.config.npars,)


def test_mle_fit_return_fitted_val(tmpdir, hypotest_args):
    """
    Check that the return structure of pyhf.infer.mle.fit with the
    return_fitted_val keyword arg is as expected
    """
    tb = pyhf.tensorlib

    _, data, model = hypotest_args
    kwargs = {"return_fitted_val": True}
    result = pyhf.infer.mle.fit(data, model, **kwargs)
    # bestfit_pars, twice_nll
    assert pyhf.tensorlib.shape(result[0]) == (model.config.npars,)
    assert isinstance(result[0], type(tb.astensor(result[0])))
    assert pyhf.tensorlib.shape(result[1]) == ()


def test_hypotest_default(tmpdir, hypotest_args):
    """
    Check that the default return structure of pyhf.infer.hypotest is as expected
    """
    tb = pyhf.tensorlib

    kwargs = {}
    result = pyhf.infer.hypotest(*hypotest_args, **kwargs)
    # CLs_obs
    assert pyhf.tensorlib.shape(result) == ()
    assert isinstance(result, type(tb.astensor(result)))


def test_hypotest_poi_outofbounds(tmpdir, hypotest_args):
    """
    Check that the fit errors for POI outside of parameter bounds
    """
    pdf = pyhf.simplemodels.hepdata_like(
        signal_data=[12.0, 11.0], bkg_data=[50.0, 52.0], bkg_uncerts=[3.0, 7.0]
    )
    data = [51, 48] + pdf.config.auxdata

    with pytest.raises(ValueError):
        pyhf.infer.hypotest(-1.0, data, pdf)

    with pytest.raises(ValueError):
        pyhf.infer.hypotest(10.1, data, pdf)


@pytest.mark.parametrize('test_stat', ['q0', 'q', 'qtilde'])
def test_hypotest_return_tail_probs(tmpdir, hypotest_args, test_stat):
    """
    Check that the return structure of pyhf.infer.hypotest with the
    return_tail_probs keyword arg is as expected
    """
    tb = pyhf.tensorlib

    kwargs = {'return_tail_probs': True, 'test_stat': test_stat}
    result = pyhf.infer.hypotest(*hypotest_args, **kwargs)
    # CLs_obs, [CL_sb, CL_b]
    assert len(list(result)) == 2
    assert isinstance(result[0], type(tb.astensor(result[0])))
    assert len(result[1]) == 1 if test_stat == 'q0' else 2
    assert check_uniform_type(result[1])


@pytest.mark.parametrize('test_stat', ['q0', 'q', 'qtilde'])
def test_hypotest_return_expected(tmpdir, hypotest_args, test_stat):
    """
    Check that the return structure of pyhf.infer.hypotest with the
    additon of the return_expected keyword arg is as expected
    """
    tb = pyhf.tensorlib

    kwargs = {
        'return_tail_probs': True,
        'return_expected': True,
        'test_stat': test_stat,
    }
    result = pyhf.infer.hypotest(*hypotest_args, **kwargs)
    # CLs_obs, [CLsb, CLb], CLs_exp
    assert len(list(result)) == 3
    assert isinstance(result[0], type(tb.astensor(result[0])))
    assert len(result[1]) == 1 if test_stat == 'q0' else 2
    assert check_uniform_type(result[1])
    assert isinstance(result[2], type(tb.astensor(result[2])))


@pytest.mark.parametrize('test_stat', ['q0', 'q', 'qtilde'])
def test_hypotest_return_expected_set(tmpdir, hypotest_args, test_stat):
    """
    Check that the return structure of pyhf.infer.hypotest with the
    additon of the return_expected_set keyword arg is as expected
    """
    tb = pyhf.tensorlib

    kwargs = {
        'return_tail_probs': True,
        'return_expected': True,
        'return_expected_set': True,
        'test_stat': test_stat,
    }
    result = pyhf.infer.hypotest(*hypotest_args, **kwargs)
    # CLs_obs, [CLsb, CLb], CLs_exp, CLs_exp @[-2, -1, 0, +1, +2]sigma
    assert len(list(result)) == 4
    assert isinstance(result[0], type(tb.astensor(result[0])))
    assert len(result[1]) == 1 if test_stat == 'q0' else 2
    assert check_uniform_type(result[1])
    assert isinstance(result[2], type(tb.astensor(result[2])))
    assert len(result[3]) == 5
    assert check_uniform_type(result[3])


@pytest.mark.parametrize(
    "kwargs",
    [{'calctype': 'asymptotics'}, {'calctype': 'toybased', 'ntoys': 5}],
    ids=lambda x: x['calctype'],
)
def test_hypotest_backends(backend, kwargs):
    """
    Check that hypotest runs fully across all backends for all calculator types.
    """
    pdf = pyhf.simplemodels.hepdata_like(
        signal_data=[12.0, 11.0], bkg_data=[50.0, 52.0], bkg_uncerts=[3.0, 7.0]
    )
    data = [51, 48] + pdf.config.auxdata
    assert pyhf.infer.hypotest(1.0, data, pdf, **kwargs) is not None


def test_inferapi_pyhf_independence():
    """
    pyhf.infer should eventually be factored out so it should be
    infependent from pyhf internals. This is testing that
    a much simpler model still can run through pyhf.infer.hypotest
    """
    from pyhf import get_backend

    class _NonPyhfConfig:
        def __init__(self):
            self.poi_index = 0
            self.npars = 2

        def suggested_init(self):
            return [1.0, 1.0]

        def suggested_bounds(self):
            return [[0.0, 10.0], [0.0, 10.0]]

        def suggested_fixed(self):
            return [False, False]

    class NonPyhfModel:
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

    assert np.isclose(cls, 0.7267836451638846)


@pytest.mark.parametrize("test_stat", ["qtilde", "q"])
def test_calculator_distributions_without_teststatistic(test_stat):
    calc = pyhf.infer.calculators.AsymptoticCalculator(
        [0.0], {}, [1.0], [(0.0, 10.0)], [False], test_stat=test_stat
    )
    with pytest.raises(RuntimeError):
        calc.distributions(1.0)


@pytest.mark.parametrize(
    "nsigma,expected_pval",
    [
        # values tabulated using ROOT.RooStats.SignificanceToPValue
        # they are consistent with relative difference < 1e-14 with scipy.stats.norm.sf
        (5, 2.866515718791945e-07),
        (6, 9.865876450377018e-10),
        (7, 1.279812543885835e-12),
        (8, 6.220960574271829e-16),
        (9, 1.1285884059538408e-19),
    ],
)
def test_asymptotic_dist_low_pvalues(backend, nsigma, expected_pval):
    rtol = 1e-8
    if backend[0].precision != '64b':
        rtol = 1e-5
    dist = pyhf.infer.calculators.AsymptoticTestStatDistribution(0)
    assert np.isclose(np.array(dist.pvalue(nsigma)), expected_pval, rtol=rtol, atol=0)


def test_significance_to_pvalue_roundtrip(backend):
    rtol = 1e-15
    if backend[0].precision != '64b':
        rtol = 1e-6
    sigma = np.arange(0, 10, 0.1)
    dist = pyhf.infer.calculators.AsymptoticTestStatDistribution(0)
    pvalue = dist.pvalue(pyhf.tensorlib.astensor(sigma))
    back_to_sigma = -scipy.stats.norm.ppf(np.array(pvalue))
    assert np.allclose(sigma, back_to_sigma, atol=0, rtol=rtol)


def test_emperical_distribution(tmpdir, hypotest_args):
    """
    Check that the empirical distribution of the test statistic gives
    expected results
    """
    tb = pyhf.tensorlib
    np.random.seed(0)

    mu_test, data, model = hypotest_args
    init_pars = model.config.suggested_init()
    par_bounds = model.config.suggested_bounds()
    fixed_params = model.config.suggested_fixed()
    pdf = model.make_pdf(tb.astensor(model.config.suggested_init()))
    samples = pdf.sample((10,))
    test_stat_dist = pyhf.infer.calculators.EmpiricalDistribution(
        tb.astensor(
            [
                pyhf.infer.test_statistics.qmu_tilde(
                    mu_test, sample, model, init_pars, par_bounds, fixed_params
                )
                for sample in samples
            ]
        )
    )

    assert test_stat_dist.samples.tolist() == pytest.approx(
        [
            0.0,
            0.13298508544818333,
            0.0,
            0.7718561591684079,
            1.8148846519001722,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0658664357855514,
        ],
        rel=1e-06,
    )
    assert test_stat_dist.pvalue(test_stat_dist.samples[4]) == 0.1
    assert test_stat_dist.expected_value(nsigma=2) == pytest.approx(
        1.6013233293819484, rel=1e-07
    )


def test_toy_calculator(tmpdir, hypotest_args):
    """
    Check that the toy calculator is performing as expected
    """
    np.random.seed(0)
    mu_test, data, model = hypotest_args
    toy_calculator_qtilde_mu = pyhf.infer.calculators.ToyCalculator(
        data, model, None, None, ntoys=10, track_progress=False
    )
    qtilde_mu_sig, qtilde_mu_bkg = toy_calculator_qtilde_mu.distributions(mu_test)
    assert qtilde_mu_sig.samples.tolist() == pytest.approx(
        [
            0.0,
            0.13298508544818333,
            0.0,
            0.7718561591684079,
            1.8148846519001722,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0658664357855514,
        ],
        rel=1e-05,
    )
    assert qtilde_mu_bkg.samples.tolist() == pytest.approx(
        [
            2.2664625678345374,
            1.0816609064021918,
            2.757021893273645,
            1.3835695381553137,
            0.47074742519123447,
            0.0,
            3.716648363429215,
            3.8021895772130847,
            5.114135203934609,
            1.3511153719425124,
        ],
        rel=1e-05,
    )
    assert toy_calculator_qtilde_mu.teststatistic(mu_test) == pytest.approx(
        3.938245064259945, rel=1e-07
    )
