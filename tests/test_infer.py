import logging
import types
import warnings

import numpy as np
import pytest
import scipy.stats

import pyhf


@pytest.fixture
def hypotest_args():
    pdf = pyhf.simplemodels.uncorrelated_background(
        signal=[12.0, 11.0], bkg=[50.0, 52.0], bkg_uncertainty=[3.0, 7.0]
    )
    mu_test = 1.0
    data = [51, 48] + pdf.config.auxdata
    return mu_test, data, pdf


def check_uniform_type(in_list):
    return all(
        isinstance(item, type(pyhf.tensorlib.astensor(item))) for item in in_list
    )


@pytest.mark.usefixtures("tmp_path")
def test_toms748_scan(hypotest_args):
    """
    Test the upper limit toms748 scan returns the correct structure and values
    """
    _, data, model = hypotest_args
    results = pyhf.infer.intervals.upper_limits.toms748_scan(
        data, model, 0, 5, rtol=1e-8
    )
    assert len(results) == 2
    observed_limit, expected_limits = results
    observed_cls = pyhf.infer.hypotest(
        observed_limit,
        data,
        model,
        model.config.suggested_init(),
        model.config.suggested_bounds(),
    )
    expected_cls = np.array(
        [
            pyhf.infer.hypotest(
                expected_limits[i],
                data,
                model,
                model.config.suggested_init(),
                model.config.suggested_bounds(),
                return_expected_set=True,
            )[1][i]
            for i in range(5)
        ]
    )
    assert observed_cls == pytest.approx(0.05)
    assert expected_cls == pytest.approx(0.05)


def test_toms748_scan_bounds_extension(hypotest_args):
    """
    Test the upper limit toms748 scan bounds can correctly extend to bracket the CLs level
    """
    _, data, model = hypotest_args
    results_default = pyhf.infer.intervals.upper_limits.toms748_scan(
        data, model, 0, 5, rtol=1e-8
    )
    observed_limit_default, expected_limits_default = results_default

    # Force bounds_low to expand
    observed_limit, expected_limits = pyhf.infer.intervals.upper_limits.toms748_scan(
        data, model, 3, 5, rtol=1e-8
    )

    assert observed_limit == pytest.approx(observed_limit_default)
    assert np.allclose(np.asarray(expected_limits), np.asarray(expected_limits_default))

    # Force bounds_up to expand
    observed_limit, expected_limits = pyhf.infer.intervals.upper_limits.toms748_scan(
        data, model, 0, 1, rtol=1e-8
    )
    assert observed_limit == pytest.approx(observed_limit_default)
    assert np.allclose(np.asarray(expected_limits), np.asarray(expected_limits_default))


def test_upper_limit_against_auto(hypotest_args):
    """
    Test upper_limit linear scan and toms748_scan return similar results
    """
    _, data, model = hypotest_args
    results_auto = pyhf.infer.intervals.upper_limits.toms748_scan(
        data, model, 0, 5, rtol=1e-3
    )
    obs_auto, exp_auto = results_auto
    results_linear = pyhf.infer.intervals.upper_limits.upper_limit(
        data, model, scan=np.linspace(0, 5, 21)
    )
    obs_linear, exp_linear = results_linear
    # Can't expect these to be much closer given the low granularity of the linear scan
    assert obs_auto == pytest.approx(obs_linear, abs=0.1)
    assert np.allclose(exp_auto, exp_linear, atol=0.1)


def test_upper_limit(hypotest_args):
    """
    Check that the default return structure of pyhf.infer.hypotest is as expected
    """
    _, data, model = hypotest_args
    scan = np.linspace(0, 5, 11)
    results = pyhf.infer.intervals.upper_limits.upper_limit(data, model, scan=scan)
    assert len(results) == 2
    observed_limit, expected_limits = results
    assert observed_limit == pytest.approx(1.0262704738584554)
    assert expected_limits == pytest.approx(
        [0.65765653, 0.87999725, 1.12453992, 1.50243428, 2.09232927]
    )

    # tighter relative tolerance needed for macos
    results = pyhf.infer.intervals.upper_limits.upper_limit(data, model, rtol=1e-6)
    assert len(results) == 2
    observed_limit, expected_limits = results
    assert observed_limit == pytest.approx(1.01156939)
    assert expected_limits == pytest.approx(
        [0.55988001, 0.75702336, 1.06234693, 1.50116923, 2.05078596]
    )


def test_upper_limit_with_kwargs(hypotest_args):
    """
    Check that the default return structure of pyhf.infer.hypotest is as expected
    """
    _, data, model = hypotest_args
    scan = np.linspace(0, 5, 11)
    results = pyhf.infer.intervals.upper_limits.upper_limit(
        data, model, scan=scan, test_stat="qtilde"
    )
    assert len(results) == 2
    observed_limit, expected_limits = results
    assert observed_limit == pytest.approx(1.0262704738584554)
    assert expected_limits == pytest.approx(
        [0.65765653, 0.87999725, 1.12453992, 1.50243428, 2.09232927]
    )

    # linear_grid_scan
    results = pyhf.infer.intervals.upper_limits.upper_limit(
        data, model, scan=scan, return_results=True
    )
    assert len(results) == 3
    observed_limit, expected_limits, (_scan, point_results) = results
    assert observed_limit == pytest.approx(1.0262704738584554)
    assert expected_limits == pytest.approx(
        [0.65765653, 0.87999725, 1.12453992, 1.50243428, 2.09232927]
    )
    assert _scan.tolist() == scan.tolist()
    assert len(_scan) == len(point_results)

    # toms748_scan
    results = pyhf.infer.intervals.upper_limits.upper_limit(
        data, model, return_results=True, rtol=1e-6
    )
    assert len(results) == 3
    observed_limit, expected_limits, (_scan, point_results) = results
    assert observed_limit == pytest.approx(1.01156939)
    assert expected_limits == pytest.approx(
        [0.55988001, 0.75702336, 1.06234693, 1.50116923, 2.05078596]
    )


def test_upper_limit_return_tail_probs(hypotest_args):
    """
    Regression test for https://github.com/scikit-hep/pyhf/issues/2669.
    upper_limit should support return_tail_probs=True without raising IndexError.
    The tail probability information should be accessible in the per-point results.
    """
    _, data, model = hypotest_args
    scan = np.linspace(0, 5, 11)

    # linear_grid_scan: return_tail_probs=True alone
    results = pyhf.infer.intervals.upper_limits.upper_limit(
        data, model, scan=scan, return_tail_probs=True
    )
    assert len(results) == 2
    observed_limit, expected_limits = results
    assert observed_limit == pytest.approx(1.0262704738584554)
    assert expected_limits == pytest.approx(
        [0.65765653, 0.87999725, 1.12453992, 1.50243428, 2.09232927]
    )

    # linear_grid_scan: return_tail_probs=True with return_results=True
    results = pyhf.infer.intervals.upper_limits.upper_limit(
        data, model, scan=scan, return_results=True, return_tail_probs=True
    )
    assert len(results) == 3
    observed_limit, expected_limits, (_scan, point_results) = results
    assert observed_limit == pytest.approx(1.0262704738584554)
    assert len(_scan) == len(point_results)
    # Each per-point result has 3 elements: (CLs, [CLsb, CLb], exp_band)
    assert len(point_results[0]) == 3

    # toms748_scan: return_tail_probs=True
    results = pyhf.infer.intervals.upper_limits.upper_limit(
        data, model, return_tail_probs=True, rtol=1e-4
    )
    assert len(results) == 2
    observed_limit, expected_limits = results
    assert observed_limit == pytest.approx(1.01156939, abs=0.1)


@pytest.mark.usefixtures("tmp_path")
def test_mle_fit_default(hypotest_args):
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


@pytest.mark.usefixtures("tmp_path")
def test_mle_fit_return_fitted_val(hypotest_args):
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


@pytest.mark.usefixtures("tmp_path")
def test_hypotest_default(hypotest_args):
    """
    Check that the default return structure of pyhf.infer.hypotest is as expected
    """
    tb = pyhf.tensorlib

    kwargs = {}
    result = pyhf.infer.hypotest(*hypotest_args, **kwargs)
    # CLs_obs
    assert pyhf.tensorlib.shape(result) == ()
    assert isinstance(result, type(tb.astensor(result)))


@pytest.mark.usefixtures("tmp_path", "hypotest_args")
def test_hypotest_poi_outofbounds():
    """
    Check that the fit errors for POI outside of parameter bounds
    """
    pdf = pyhf.simplemodels.uncorrelated_background(
        signal=[12.0, 11.0], bkg=[50.0, 52.0], bkg_uncertainty=[3.0, 7.0]
    )
    data = [51, 48] + pdf.config.auxdata

    with pytest.raises(
        ValueError, match=r"fit initialization parameter.+lies outside of its bounds"
    ):
        pyhf.infer.hypotest(-1.0, data, pdf)

    with pytest.raises(
        ValueError, match=r"fit initialization parameter.+lies outside of its bounds"
    ):
        pyhf.infer.hypotest(10.1, data, pdf)


@pytest.mark.parametrize("test_stat", ["q0", "q", "qtilde"])
@pytest.mark.usefixtures("tmp_path")
def test_hypotest_return_tail_probs(hypotest_args, test_stat):
    """
    Check that the return structure of pyhf.infer.hypotest with the
    return_tail_probs keyword arg is as expected
    """
    tb = pyhf.tensorlib

    kwargs = {"return_tail_probs": True, "test_stat": test_stat}
    result = pyhf.infer.hypotest(*hypotest_args, **kwargs)
    # CLs_obs, [CL_sb, CL_b]
    assert len(list(result)) == 2
    assert isinstance(result[0], type(tb.astensor(result[0])))
    assert len(result[1]) == 1 if test_stat == "q0" else 2
    assert check_uniform_type(result[1])


@pytest.mark.parametrize("test_stat", ["q0", "q", "qtilde"])
@pytest.mark.usefixtures("tmp_path")
def test_hypotest_return_expected(hypotest_args, test_stat):
    """
    Check that the return structure of pyhf.infer.hypotest with the
    addition of the return_expected keyword arg is as expected
    """
    tb = pyhf.tensorlib

    kwargs = {
        "return_tail_probs": True,
        "return_expected": True,
        "test_stat": test_stat,
    }
    result = pyhf.infer.hypotest(*hypotest_args, **kwargs)
    # CLs_obs, [CLsb, CLb], CLs_exp
    assert len(list(result)) == 3
    assert isinstance(result[0], type(tb.astensor(result[0])))
    assert len(result[1]) == 1 if test_stat == "q0" else 2
    assert check_uniform_type(result[1])
    assert isinstance(result[2], type(tb.astensor(result[2])))


@pytest.mark.parametrize("test_stat", ["q0", "q", "qtilde"])
@pytest.mark.usefixtures("tmp_path")
def test_hypotest_return_expected_set(hypotest_args, test_stat):
    """
    Check that the return structure of pyhf.infer.hypotest with the
    addition of the return_expected_set keyword arg is as expected
    """
    tb = pyhf.tensorlib

    kwargs = {
        "return_tail_probs": True,
        "return_expected": True,
        "return_expected_set": True,
        "test_stat": test_stat,
    }
    result = pyhf.infer.hypotest(*hypotest_args, **kwargs)
    # CLs_obs, [CLsb, CLb], CLs_exp, CLs_exp @[-2, -1, 0, +1, +2]sigma
    assert len(list(result)) == 4
    assert isinstance(result[0], type(tb.astensor(result[0])))
    assert len(result[1]) == 1 if test_stat == "q0" else 2
    assert check_uniform_type(result[1])
    assert isinstance(result[2], type(tb.astensor(result[2])))
    assert len(result[3]) == 5
    assert check_uniform_type(result[3])


@pytest.mark.parametrize(
    ("calctype", "kwargs", "expected_type"),
    [
        ("asymptotics", {}, pyhf.infer.calculators.AsymptoticCalculator),
        ("toybased", {"ntoys": 1}, pyhf.infer.calculators.ToyCalculator),
    ],
)
@pytest.mark.parametrize("return_tail_probs", [True, False])
@pytest.mark.parametrize("return_expected", [True, False])
@pytest.mark.parametrize("return_expected_set", [True, False])
@pytest.mark.usefixtures("tmp_path")
def test_hypotest_return_calculator(
    hypotest_args,
    calctype,
    kwargs,
    expected_type,
    return_tail_probs,
    return_expected,
    return_expected_set,
):
    """
    Check that the return structure of pyhf.infer.hypotest with the
    addition of the return_calculator keyword arg is as expected
    """
    *_, _model = hypotest_args

    # only those return flags where the toggled return value
    # is placed in front of the calculator in the returned tuple
    extra_returns = sum(
        int(return_flag)
        for return_flag in (
            return_tail_probs,
            return_expected,
            return_expected_set,
        )
    )

    result = pyhf.infer.hypotest(
        *hypotest_args,
        return_calculator=True,
        return_tail_probs=return_tail_probs,
        return_expected=return_expected,
        return_expected_set=return_expected_set,
        calctype=calctype,
        **kwargs,
    )

    assert len(list(result)) == 2 + extra_returns
    # not *_, calc = result b.c. in future, there could be additional optional returns
    calc = result[1 + extra_returns]
    assert isinstance(calc, expected_type)


@pytest.mark.parametrize(
    "kwargs",
    [{"calctype": "asymptotics"}, {"calctype": "toybased", "ntoys": 5}],
    ids=lambda x: x["calctype"],
)
@pytest.mark.usefixtures("backend")
def test_hypotest_backends(kwargs):
    """
    Check that hypotest runs fully across all backends for all calculator types.
    """
    pdf = pyhf.simplemodels.uncorrelated_background(
        signal=[12.0, 11.0], bkg=[50.0, 52.0], bkg_uncertainty=[3.0, 7.0]
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
            _, gamma = pars
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


def test_clipped_normal_calc(hypotest_args):
    mu_test, data, pdf = hypotest_args
    _, expected_clipped_normal = pyhf.infer.hypotest(
        mu_test,
        data,
        pdf,
        return_expected_set=True,
        calc_base_dist="clipped_normal",
    )
    _, expected_normal = pyhf.infer.hypotest(
        mu_test,
        data,
        pdf,
        return_expected_set=True,
        calc_base_dist="normal",
    )
    assert expected_clipped_normal[-1] < expected_normal[-1]

    with pytest.raises(ValueError, match="unknown base distribution for asymptotics"):
        pyhf.infer.hypotest(
            mu_test,
            data,
            pdf,
            return_expected_set=True,
            calc_base_dist="unknown",
        )


@pytest.mark.parametrize("test_stat", ["qtilde", "q"])
def test_calculator_distributions_without_teststatistic(test_stat):
    calc = pyhf.infer.calculators.AsymptoticCalculator(
        [0.0], {}, [1.0], [(0.0, 10.0)], [False], test_stat=test_stat
    )
    with pytest.raises(RuntimeError):
        calc.distributions(1.0)


@pytest.mark.parametrize("calculator", ["AsymptoticCalculator", "ToyCalculator"])
def test_calculator_distributions_poi_test_keyword(calculator):
    # Regression test for PR #2688: ``poi_test`` must stay a public keyword
    model = pyhf.simplemodels.uncorrelated_background(
        signal=[12.0, 11.0], bkg=[50.0, 52.0], bkg_uncertainty=[3.0, 7.0]
    )
    data = [51, 48] + model.config.auxdata
    kwargs = {"ntoys": 10} if calculator == "ToyCalculator" else {}
    calc = getattr(pyhf.infer.calculators, calculator)(data, model, **kwargs)
    calc.teststatistic(poi_test=1.0)
    sig_plus_bkg_dist, bkg_dist = calc.distributions(poi_test=1.0)
    assert sig_plus_bkg_dist is not None
    assert bkg_dist is not None


@pytest.mark.parametrize(
    ("nsigma", "expected_pval"),
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
    if backend[0].precision != "64b":
        rtol = 1e-5
    dist = pyhf.infer.calculators.AsymptoticTestStatDistribution(0)
    assert np.isclose(np.array(dist.pvalue(nsigma)), expected_pval, rtol=rtol, atol=0)


def test_significance_to_pvalue_roundtrip(backend):
    rtol = 1e-15
    if backend[0].precision != "64b":
        rtol = 1e-6
    sigma = np.arange(0, 10, 0.1)
    dist = pyhf.infer.calculators.AsymptoticTestStatDistribution(0)
    pvalue = dist.pvalue(pyhf.tensorlib.astensor(sigma))
    back_to_sigma = -scipy.stats.norm.ppf(np.array(pvalue))
    assert np.allclose(sigma, back_to_sigma, atol=0, rtol=rtol)


@pytest.mark.usefixtures("tmp_path")
def test_emperical_distribution(hypotest_args):
    """
    Check that the empirical distribution of the test statistic gives
    expected results
    """
    tb = pyhf.tensorlib
    np.random.seed(0)

    mu_test, _, model = hypotest_args
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
            0.13298492825293806,
            0.0,
            0.7718560148925349,
            1.814884694401428,
            0.0,
            0.0,
            0.0,
            0.0,
            0.06586643485326249,
        ],
        1e-07,
    )
    assert test_stat_dist.pvalue(test_stat_dist.samples[4]) == 0.1
    assert test_stat_dist.expected_value(nsigma=2) == pytest.approx(
        1.6013233336403654, 1e-07
    )


@pytest.mark.usefixtures("tmp_path")
def test_toy_calculator(hypotest_args):
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
            0.017350013494649374,
            0.0,
            0.2338008822475217,
            0.020328779776718875,
            0.8911134903562186,
            0.04408274703718007,
            0.0,
            0.03977591672014569,
            0.0,
        ],
        1e-07,
    )
    assert qtilde_mu_bkg.samples.tolist() == pytest.approx(
        [
            5.642956861215396,
            0.37581364290284114,
            4.875367689039649,
            3.4299006094989295,
            1.0161021805475343,
            0.03345317321810626,
            0.21984803001140563,
            1.274869119189077,
            9.368264062021098,
            3.0716486684082156,
        ],
        1e-07,
    )
    assert toy_calculator_qtilde_mu.teststatistic(mu_test) == pytest.approx(
        3.938244920380498, 1e-07
    )


@pytest.fixture
def failing_toys(monkeypatch):
    """
    Make the qtilde test statistic raise FailedMinimization on the given 0-indexed
    call indices. ToyCalculator.distributions calls it once per toy, signal-like
    toys first, so with ntoys=10 the calls 0-9 are the signal-like toys and the
    calls 10-19 the background-like toys.
    """
    original_func = pyhf.infer.test_statistics.qmu_tilde

    def inject(fail_on):
        state = types.SimpleNamespace(
            n_calls=0,
            fake_result=types.SimpleNamespace(
                success=False, message="simulated failure"
            ),
        )

        def patched_teststat(*args, **kwargs):
            idx = state.n_calls
            state.n_calls += 1
            if idx in fail_on:
                raise pyhf.exceptions.FailedMinimization(state.fake_result)
            return original_func(*args, **kwargs)

        monkeypatch.setattr(pyhf.infer.utils, "qmu_tilde", patched_teststat)
        return state

    return inject


def test_toy_calculator_toy_results_no_failures(hypotest_args, caplog):
    """
    Check that toy_results is populated after distributions() and reflects
    all-successful toys when no failures occur.
    """
    np.random.seed(0)
    mu_test, data, model = hypotest_args
    calc = pyhf.infer.calculators.ToyCalculator(
        data, model, ntoys=10, track_progress=False
    )
    assert calc.toy_results is None

    with caplog.at_level(logging.DEBUG, logger="pyhf.infer.calculators"):
        calc.distributions(mu_test)

    assert set(calc.toy_results) == {"sig_plus_bkg", "bkg_only"}

    sig_result = calc.toy_results["sig_plus_bkg"]
    bkg_result = calc.toy_results["bkg_only"]

    assert isinstance(sig_result, pyhf.infer.calculators.ToyResult)
    assert isinstance(bkg_result, pyhf.infer.calculators.ToyResult)

    assert len(sig_result.successful) == 10
    assert len(sig_result.failed) == 0
    assert len(bkg_result.successful) == 10
    assert len(bkg_result.failed) == 0

    assert not [r for r in caplog.records if r.name == "pyhf.infer.calculators"]


def test_toy_calculator_default_raises_on_failure(hypotest_args, failing_toys):
    """
    With failure_threshold=None (default), any FailedMinimization propagates
    immediately, and the failed toy is still recorded in toy_results.
    """
    np.random.seed(0)
    mu_test, data, model = hypotest_args
    calc = pyhf.infer.calculators.ToyCalculator(
        data, model, ntoys=10, track_progress=False
    )
    state = failing_toys({2})

    with pytest.raises(pyhf.exceptions.FailedMinimization) as excinfo:
        calc.distributions(mu_test)

    # The default behaviour re-raises the original exception unchanged
    assert type(excinfo.value) is pyhf.exceptions.FailedMinimization
    assert excinfo.value.result is state.fake_result
    assert state.n_calls == 3

    assert set(calc.toy_results) == {"sig_plus_bkg"}
    sig_result = calc.toy_results["sig_plus_bkg"]
    assert [toy_idx for toy_idx, _, _ in sig_result.failed] == [2]
    assert len(sig_result.successful) == 2


def test_toy_calculator_failure_threshold_allows_failures(hypotest_args, failing_toys):
    """
    With failure_threshold set, toys that fail are recorded in toy_results and
    distributions() completes with the test statistics of the successful toys.
    """
    np.random.seed(0)
    mu_test, data, model = hypotest_args
    calc = pyhf.infer.calculators.ToyCalculator(
        data, model, ntoys=10, track_progress=False, failure_threshold=0.5
    )
    state = failing_toys({2, 5})

    sig_dist, bkg_dist = calc.distributions(mu_test)

    sig_result = calc.toy_results["sig_plus_bkg"]
    bkg_result = calc.toy_results["bkg_only"]

    assert len(sig_result.failed) == 2
    assert len(sig_result.successful) == 8
    assert len(bkg_result.failed) == 0
    assert len(bkg_result.successful) == 10

    # Each failed entry is (toy_index, sample, exception)
    assert [toy_idx for toy_idx, _, _ in sig_result.failed] == [2, 5]
    for _, sample, exc in sig_result.failed:
        assert len(sample) == len(data)
        assert isinstance(exc, pyhf.exceptions.FailedMinimization)
        assert exc.result is state.fake_result

    # The values of test_toy_calculator without the failed toys 2 and 5
    assert sig_dist.samples.tolist() == pytest.approx(
        [
            0.0,
            0.017350013494649374,
            0.2338008822475217,
            0.020328779776718875,
            0.04408274703718007,
            0.0,
            0.03977591672014569,
            0.0,
        ],
        1e-07,
    )
    assert bkg_dist.samples.tolist() == pytest.approx(
        [
            5.642956861215396,
            0.37581364290284114,
            4.875367689039649,
            3.4299006094989295,
            1.0161021805475343,
            0.03345317321810626,
            0.21984803001140563,
            1.274869119189077,
            9.368264062021098,
            3.0716486684082156,
        ],
        1e-07,
    )


def test_toy_calculator_background_like_failures(hypotest_args, failing_toys):
    """
    Failures in the background-like loop are recorded under "bkg_only" with toy
    indices relative to that loop, and the signal-like toys are unaffected.
    """
    np.random.seed(0)
    mu_test, data, model = hypotest_args
    calc = pyhf.infer.calculators.ToyCalculator(
        data, model, ntoys=10, track_progress=False, failure_threshold=0.2
    )
    failing_toys({10, 13})

    sig_dist, bkg_dist = calc.distributions(mu_test)

    sig_result = calc.toy_results["sig_plus_bkg"]
    bkg_result = calc.toy_results["bkg_only"]

    assert len(sig_result.failed) == 0
    assert len(sig_result.successful) == 10
    assert [toy_idx for toy_idx, _, _ in bkg_result.failed] == [0, 3]
    assert len(bkg_result.successful) == 8

    assert sig_dist.samples.tolist() == pytest.approx(
        [
            0.0,
            0.017350013494649374,
            0.0,
            0.2338008822475217,
            0.020328779776718875,
            0.8911134903562186,
            0.04408274703718007,
            0.0,
            0.03977591672014569,
            0.0,
        ],
        1e-07,
    )
    # The values of test_toy_calculator without the failed toys 0 and 3
    assert bkg_dist.samples.tolist() == pytest.approx(
        [
            0.37581364290284114,
            4.875367689039649,
            1.0161021805475343,
            0.03345317321810626,
            0.21984803001140563,
            1.274869119189077,
            9.368264062021098,
            3.0716486684082156,
        ],
        1e-07,
    )


def test_toy_calculator_failure_threshold_per_hypothesis(hypotest_args, failing_toys):
    """
    The failure_threshold is applied to the signal-like and background-like toys
    separately, so one failure in each loop is allowed with failure_threshold=0.1.
    """
    np.random.seed(0)
    mu_test, data, model = hypotest_args
    calc = pyhf.infer.calculators.ToyCalculator(
        data, model, ntoys=10, track_progress=False, failure_threshold=0.1
    )
    failing_toys({4, 13})

    sig_dist, bkg_dist = calc.distributions(mu_test)

    assert [idx for idx, _, _ in calc.toy_results["sig_plus_bkg"].failed] == [4]
    assert [idx for idx, _, _ in calc.toy_results["bkg_only"].failed] == [3]
    assert len(sig_dist.samples) == 9
    assert len(bkg_dist.samples) == 9


def test_toy_calculator_failure_on_first_toy_allowed(hypotest_args, failing_toys):
    """
    The failure_threshold is a fraction of ntoys, not of the toys run so far, so a
    failure on the very first toy is within failure_threshold=0.1 for ntoys=10.
    """
    np.random.seed(0)
    mu_test, data, model = hypotest_args
    calc = pyhf.infer.calculators.ToyCalculator(
        data, model, ntoys=10, track_progress=False, failure_threshold=0.1
    )
    failing_toys({0})

    sig_dist, _ = calc.distributions(mu_test)

    sig_result = calc.toy_results["sig_plus_bkg"]
    assert [toy_idx for toy_idx, _, _ in sig_result.failed] == [0]
    assert len(sig_result.successful) == 9
    assert len(sig_dist.samples) == 9


@pytest.mark.parametrize(
    ("failure_threshold", "n_failures"),
    [(0.1, 1), (0.2, 2), (0.3, 3)],
    ids=["0.1-1", "0.2-2", "0.3-3"],
)
def test_toy_calculator_failure_threshold_boundary_allowed(
    hypotest_args, failing_toys, failure_threshold, n_failures
):
    """
    A number of failures exactly equal to failure_threshold * ntoys is allowed.
    """
    np.random.seed(0)
    mu_test, data, model = hypotest_args
    calc = pyhf.infer.calculators.ToyCalculator(
        data,
        model,
        ntoys=10,
        track_progress=False,
        failure_threshold=failure_threshold,
    )
    failing_toys(set(range(n_failures)))

    sig_dist, _ = calc.distributions(mu_test)

    assert len(calc.toy_results["sig_plus_bkg"].failed) == n_failures
    assert len(sig_dist.samples) == 10 - n_failures


@pytest.mark.parametrize(
    ("failure_threshold", "n_failures"),
    [(0.0, 1), (0.1, 2), (0.2, 3)],
    ids=["0.0-1", "0.1-2", "0.2-3"],
)
def test_toy_calculator_failure_threshold_boundary_exceeded(
    hypotest_args, failing_toys, failure_threshold, n_failures
):
    """
    One failure more than failure_threshold * ntoys raises TooManyFailedToys as
    soon as it happens, without running the remaining toys.
    """
    np.random.seed(0)
    mu_test, data, model = hypotest_args
    calc = pyhf.infer.calculators.ToyCalculator(
        data,
        model,
        ntoys=10,
        track_progress=False,
        failure_threshold=failure_threshold,
    )
    state = failing_toys(set(range(n_failures)))

    with pytest.raises(pyhf.exceptions.TooManyFailedToys):
        calc.distributions(mu_test)

    assert state.n_calls == n_failures


def test_toy_calculator_failure_threshold_exceeded_raises(hypotest_args, failing_toys):
    """
    Exceeding the allowed failure fraction raises TooManyFailedToys, chained from
    the triggering FailedMinimization, and keeps the diagnostics of the toys run.
    """
    np.random.seed(0)
    mu_test, data, model = hypotest_args
    calc = pyhf.infer.calculators.ToyCalculator(
        data, model, ntoys=10, track_progress=False, failure_threshold=0.1
    )
    state = failing_toys({3, 6})

    with pytest.raises(
        pyhf.exceptions.TooManyFailedToys,
        match=r"^Signal-like: 2 of the 7 toys run failed \(ntoys=10, failure_threshold=0\.1\)\.",
    ) as excinfo:
        calc.distributions(mu_test)

    # Existing handlers for FailedMinimization keep catching it
    assert isinstance(excinfo.value, pyhf.exceptions.FailedMinimization)
    assert excinfo.value.result is state.fake_result
    assert isinstance(excinfo.value.__cause__, pyhf.exceptions.FailedMinimization)
    assert excinfo.value.__cause__.result is state.fake_result
    assert state.n_calls == 7

    assert set(calc.toy_results) == {"sig_plus_bkg"}
    sig_result = calc.toy_results["sig_plus_bkg"]
    assert [toy_idx for toy_idx, _, _ in sig_result.failed] == [3, 6]
    assert len(sig_result.successful) == 5


def test_toy_calculator_all_toys_failed_raises(hypotest_args, failing_toys):
    """
    Even with failure_threshold=1.0, at least one toy per hypothesis has to succeed,
    instead of building an empty distribution with a NaN p-value.
    """
    np.random.seed(0)
    mu_test, data, model = hypotest_args
    calc = pyhf.infer.calculators.ToyCalculator(
        data, model, ntoys=10, track_progress=False, failure_threshold=1.0
    )
    state = failing_toys(set(range(10)))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(
            pyhf.exceptions.TooManyFailedToys,
            match=r"^Signal-like: 10 of the 10 toys run failed \(ntoys=10, failure_threshold=1\.0\)\.",
        ) as excinfo:
            calc.distributions(mu_test)

    assert excinfo.value.__cause__ is calc.toy_results["sig_plus_bkg"].failed[-1][2]
    assert len(calc.toy_results["sig_plus_bkg"].failed) == 10
    assert len(calc.toy_results["sig_plus_bkg"].successful) == 0
    assert state.n_calls == 10


def test_toy_calculator_single_successful_toy(hypotest_args, failing_toys):
    """
    With failure_threshold=1.0, a single successful toy is enough to build the
    distribution.
    """
    np.random.seed(0)
    mu_test, data, model = hypotest_args
    calc = pyhf.infer.calculators.ToyCalculator(
        data, model, ntoys=10, track_progress=False, failure_threshold=1.0
    )
    failing_toys(set(range(10)) - {3})

    sig_dist, bkg_dist = calc.distributions(mu_test)

    # Toy 3 of test_toy_calculator
    assert sig_dist.samples.tolist() == pytest.approx([0.2338008822475217], 1e-07)
    assert len(bkg_dist.samples) == 10


@pytest.mark.parametrize("failure_threshold", [-0.1, 1.1, float("nan")])
def test_toy_calculator_invalid_failure_threshold(hypotest_args, failure_threshold):
    """
    A failure_threshold outside of [0.0, 1.0] is rejected.
    """
    _, data, model = hypotest_args
    with pytest.raises(ValueError, match="failure_threshold"):
        pyhf.infer.calculators.ToyCalculator(
            data, model, failure_threshold=failure_threshold
        )


@pytest.mark.parametrize("failure_threshold", [None, 0.0, 0.5, 1.0])
def test_toy_calculator_valid_failure_threshold(hypotest_args, failure_threshold):
    """
    None and any failure_threshold within [0.0, 1.0] are accepted.
    """
    _, data, model = hypotest_args
    calc = pyhf.infer.calculators.ToyCalculator(
        data, model, failure_threshold=failure_threshold
    )
    assert calc.failure_threshold == failure_threshold


def test_toy_calculator_toy_results_replaced(hypotest_args, failing_toys):
    """
    Calling distributions() again replaces the toy_results of the previous call.
    """
    np.random.seed(0)
    mu_test, data, model = hypotest_args
    calc = pyhf.infer.calculators.ToyCalculator(
        data, model, ntoys=10, track_progress=False, failure_threshold=0.5
    )
    # Only the first call to distributions() has a failed toy
    failing_toys({2})

    calc.distributions(mu_test)
    assert len(calc.toy_results["sig_plus_bkg"].failed) == 1

    calc.distributions(mu_test)
    assert len(calc.toy_results["sig_plus_bkg"].failed) == 0
    assert len(calc.toy_results["sig_plus_bkg"].successful) == 10
    assert len(calc.toy_results["bkg_only"].successful) == 10


def test_toy_calculator_failure_warning_logged(hypotest_args, failing_toys, caplog):
    """
    When toys fail within the threshold, a warning is logged with the failure count.
    """
    np.random.seed(0)
    mu_test, data, model = hypotest_args
    calc = pyhf.infer.calculators.ToyCalculator(
        data, model, ntoys=10, track_progress=False, failure_threshold=0.5
    )
    failing_toys({1})

    with caplog.at_level(logging.WARNING, logger="pyhf.infer.calculators"):
        calc.distributions(mu_test)

    records = [
        (r.name, r.levelno, r.getMessage())
        for r in caplog.records
        if r.name == "pyhf.infer.calculators"
    ]
    assert records == [
        (
            "pyhf.infer.calculators",
            logging.WARNING,
            "Signal-like: 1/10 toys failed (10.0%)",
        )
    ]


def test_toy_calculator_failure_threshold_via_hypotest(hypotest_args, failing_toys):
    """
    failure_threshold is forwarded through hypotest() kwargs to ToyCalculator, and
    the diagnostics are available from the returned calculator.
    """
    np.random.seed(0)
    mu_test, data, model = hypotest_args
    # hypotest() calls teststatistic() first (call 0), so call 1 is the first
    # signal-like toy
    failing_toys({1})

    CLs, calc = pyhf.infer.hypotest(
        mu_test,
        data,
        model,
        ntoys=10,
        track_progress=False,
        calctype="toybased",
        failure_threshold=0.1,
        return_calculator=True,
    )

    assert [idx for idx, _, _ in calc.toy_results["sig_plus_bkg"].failed] == [0]
    assert len(calc.toy_results["bkg_only"].failed) == 0
    assert np.isfinite(CLs)


@pytest.mark.usefixtures("backend")
def test_toy_calculator_failures_backends(failing_toys):
    """
    Check that failed toys are handled across all backends.
    """
    pdf = pyhf.simplemodels.uncorrelated_background(
        signal=[12.0, 11.0], bkg=[50.0, 52.0], bkg_uncertainty=[3.0, 7.0]
    )
    data = [51, 48] + pdf.config.auxdata
    mu_test = 1.0
    calc = pyhf.infer.calculators.ToyCalculator(
        data, pdf, ntoys=10, track_progress=False, failure_threshold=0.5
    )
    failing_toys({0, 10})

    sig_dist, bkg_dist = calc.distributions(mu_test)

    assert len(calc.toy_results["sig_plus_bkg"].failed) == 1
    assert len(calc.toy_results["sig_plus_bkg"].successful) == 9
    assert len(calc.toy_results["bkg_only"].failed) == 1
    assert len(calc.toy_results["bkg_only"].successful) == 9
    assert len(sig_dist.samples) == 9
    assert len(bkg_dist.samples) == 9
    assert np.isfinite(sig_dist.pvalue(calc.teststatistic(mu_test)))


def test_fixed_poi(hypotest_args):
    """
    Check that the return structure of pyhf.infer.hypotest with the
    addition of the return_expected keyword arg is as expected
    """

    test_poi, data, model = hypotest_args
    model.config.param_set("mu").suggested_fixed = [True]
    with pytest.raises(pyhf.exceptions.InvalidModel):
        pyhf.infer.hypotest(test_poi, data, model)


def test_teststat_nan_guard():
    # Example from Issue #1992
    model = pyhf.simplemodels.uncorrelated_background(
        signal=[1.0], bkg=[1.0], bkg_uncertainty=[1.0]
    )
    observations = [2]
    test_poi = 0.0
    data = observations + model.config.auxdata
    init_pars = model.config.suggested_init()
    par_bounds = model.config.suggested_bounds()
    fixed_params = model.config.suggested_fixed()

    test_stat = pyhf.infer.test_statistics.qmu_tilde(
        test_poi, data, model, init_pars, par_bounds, fixed_params
    )
    assert test_stat == pytest.approx(0.0)
    asymptotic_calculator = pyhf.infer.calculators.AsymptoticCalculator(
        data, model, test_stat="qtilde"
    )
    # ensure not nan
    assert ~np.isnan(asymptotic_calculator.teststatistic(test_poi))
    assert asymptotic_calculator.teststatistic(test_poi) == pytest.approx(0.0)

    # Example from Issue #529
    model = pyhf.simplemodels.uncorrelated_background([0.005], [28.0], [5.0])
    test_poi = 1.0
    data = [28.0] + model.config.auxdata

    test_results = pyhf.infer.hypotest(
        test_poi, data, model, test_stat="qtilde", return_expected=True
    )
    assert all(~np.isnan(result) for result in test_results)


def test_teststatistic_poi0_no_divide_by_zero(hypotest_args):
    # Issue https://github.com/scikit-hep/pyhf/issues/2722
    # At poi_test == asimov_mu (poi=0 for qtilde) sqrtqmuA_v == 0,
    # results in the muhat < 0 branch of Equation 16 from
    # https://arxiv.org/abs/1007.1727 being 0/0. This only occurs when the
    # observed q̃_0 rounds to a tiny positive (happens on macOS arm64).
    # Raise as an error to ensure this guard does not regress.
    _, data, model = hypotest_args

    calc = pyhf.infer.calculators.AsymptoticCalculator(data, model, test_stat="qtilde")
    with warnings.catch_warnings():
        warnings.filterwarnings("error", "divide by zero encountered")
        teststat = calc.teststatistic(0.0)
    assert np.isfinite(np.asarray(teststat))

    # CLs at the background-only point must be finite and ~1 (mu=0 is never
    # excluded), not NaN from a +inf test statistic.
    with warnings.catch_warnings():
        warnings.filterwarnings("error", "divide by zero encountered")
        CLs_obs = pyhf.infer.hypotest(0.0, data, model, test_stat="qtilde")
    assert np.isfinite(CLs_obs)
    assert CLs_obs == pytest.approx(1.0, abs=1e-3)


# TODO: Remove after pyhf v0.9.0 is released
def test_deprecated_upperlimit(hypotest_args):
    with warnings.catch_warnings(record=True) as _warning:
        # Cause all warnings to always be triggered
        warnings.simplefilter("always")

        _, data, model = hypotest_args
        pyhf.infer.intervals.upperlimit(data, model, scan=np.linspace(0, 5, 11))
        assert len(_warning) == 1
        assert issubclass(_warning[-1].category, DeprecationWarning)
        assert (
            "pyhf.infer.intervals.upperlimit is deprecated in favor of pyhf.infer.intervals.upper_limits.upper_limit"
            in str(_warning[-1].message)
        )


def test_issue2563_toms748_scan_set_level(hypotest_args):
    """
    Test that setting the level actually gives us what we expect for scan=None (using the toms748 algorithm)
    """
    _, data, model = hypotest_args

    obs_limit_95, exp_limits_95 = pyhf.infer.intervals.upper_limits.upper_limit(
        data, model, rtol=1e-8, level=0.05, scan=None
    )

    obs_limit_90, exp_limits_90 = pyhf.infer.intervals.upper_limits.upper_limit(
        data, model, rtol=1e-8, level=0.10, scan=None
    )

    assert obs_limit_95 != pytest.approx(obs_limit_90), (
        "Observed limit at 95% is the same as the observed limit at 90%"
    )
    for index, (exp_limit_95, exp_limit_90) in enumerate(
        zip(exp_limits_95, exp_limits_90), 1
    ):
        assert exp_limit_95 != pytest.approx(exp_limit_90), (
            f"Expected limit for 95% is the same as the expected limit at 90% for position {index}"
        )
