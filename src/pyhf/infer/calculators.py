"""
Calculators for Hypothesis Testing.

The role of the calculators is to compute test statistic and
provide distributions of said test statistic under various
hypotheses.

Using the calculators hypothesis tests can then be performed.
"""
from .mle import fixed_poi_fit
from .. import get_backend
from . import utils
import tqdm

import logging

log = logging.getLogger(__name__)


def generate_asimov_data(asimov_mu, data, pdf, init_pars, par_bounds, fixed_params):
    """
    Compute Asimov Dataset (expected yields at best-fit values) for a given POI value.

    Example:

        >>> import pyhf
        >>> pyhf.set_backend("numpy")
        >>> model = pyhf.simplemodels.hepdata_like(
        ...     signal_data=[12.0, 11.0], bkg_data=[50.0, 52.0], bkg_uncerts=[3.0, 7.0]
        ... )
        >>> observations = [51, 48]
        >>> data = observations + model.config.auxdata
        >>> mu_test = 1.0
        >>> pyhf.infer.calculators.generate_asimov_data(mu_test, data, model, None, None, None)
        array([ 60.61332311,  56.52750217, 270.07401727,  48.31490026])

    Args:
        asimov_mu (:obj:`float`): The value for the parameter of interest to be used.
        data (:obj:`tensor`): The observed data.
        pdf (~pyhf.pdf.Model): The statistical model adhering to the schema ``model.json``.
        init_pars (:obj:`tensor`): The initial parameter values to be used for fitting.
        par_bounds (:obj:`tensor`): The parameter value bounds to be used for fitting.
        fixed_params (:obj:`tensor`): Parameters to be held constant in the fit.

    Returns:
        Tensor: The Asimov dataset.

    """
    bestfit_nuisance_asimov = fixed_poi_fit(
        asimov_mu, data, pdf, init_pars, par_bounds, fixed_params
    )
    return pdf.expected_data(bestfit_nuisance_asimov)


class AsymptoticTestStatDistribution:
    r"""
    The distribution the test statistic in the asymptotic case.

    Note: These distributions are in :math:`-\hat{\mu}/\sigma` space.
    In the ROOT implementation the same sigma is assumed for both hypotheses
    and :math:`p`-values etc are computed in that space.
    This assumption is necessarily valid, but we keep this for compatibility reasons.

    In the :math:`-\hat{\mu}/\sigma` space, the test statistic (i.e. :math:`\hat{\mu}/\sigma`) is
    normally distributed with unit variance and its mean at
    the :math:`-\mu'`, where :math:`\mu'` is the true poi value of the hypothesis.
    """

    def __init__(self, shift):
        """
        Asymptotic test statistic distribution.

        Args:
            shift (:obj:`float`): The displacement of the test statistic distribution.

        Returns:
            ~pyhf.infer.calculators.AsymptoticTestStatDistribution: The asymptotic distribution of test statistic.

        """
        self.shift = shift
        self.sqrtqmuA_v = None

    def cdf(self, value):
        """
        Compute the value of the cumulative distribution function for a given value of the test statistic.

        Example:

            >>> import pyhf
            >>> pyhf.set_backend("numpy")
            >>> bkg_dist = pyhf.infer.calculators.AsymptoticTestStatDistribution(0.0)
            >>> bkg_dist.pvalue(0)
            0.5

        Args:
            value (:obj:`float`): The test statistic value.

        Returns:
            Float: The integrated probability to observe a test statistic less than or equal to the observed ``value``.

        """
        tensorlib, _ = get_backend()
        return tensorlib.normal_cdf(value - self.shift)

    def pvalue(self, value):
        r"""
        The :math:`p`-value for a given value of the test statistic corresponding
        to signal strength :math:`\mu` and Asimov strength :math:`\mu'` as
        defined in Equations (59) and (57) of :xref:`arXiv:1007.1727`

        .. math::

            p_{\mu} = 1-F\left(q_{\mu}\middle|\mu'\right) = 1- \Phi\left(\sqrt{q_{\mu}} - \frac{\left(\mu-\mu'\right)}{\sigma}\right)

        with Equation (29)

        .. math::

            \frac{(\mu-\mu')}{\sigma} = \sqrt{\Lambda}= \sqrt{q_{\mu,A}}

        given the observed test statistics :math:`q_{\mu}` and :math:`q_{\mu,A}`.

        Args:
            value (:obj:`float`): The test statistic value.

        Returns:
            Float: The integrated probability to observe a value at least as large as the observed one.

        """
        tensorlib, _ = get_backend()
        # computing cdf(-x) instead of 1-cdf(x) for right-tail p-value for improved numerical stability
        return tensorlib.normal_cdf(-(value - self.shift))

    def expected_value(self, nsigma):
        """
        Return the expected value of the test statistic.

        Example:

            >>> import pyhf
            >>> pyhf.set_backend("numpy")
            >>> bkg_dist = pyhf.infer.calculators.AsymptoticTestStatDistribution(0.0)
            >>> n_sigma = pyhf.tensorlib.astensor([-2, -1, 0, 1, 2])
            >>> bkg_dist.expected_value(n_sigma)
            array([-2., -1.,  0.,  1.,  2.])

        Args:
            nsigma (:obj:`int` or :obj:`tensor`): The number of standard deviations.

        Returns:
            Float: The expected value of the test statistic.
        """
        return self.shift + nsigma


class AsymptoticCalculator:
    """The Asymptotic Calculator."""

    def __init__(
        self,
        data,
        pdf,
        init_pars=None,
        par_bounds=None,
        fixed_params=None,
        test_stat="qtilde",
    ):
        r"""
        Asymptotic Calculator.

        Args:
            data (:obj:`tensor`): The observed data.
            pdf (~pyhf.pdf.Model): The statistical model adhering to the schema ``model.json``.
            init_pars (:obj:`tensor`): The initial parameter values to be used for fitting.
            par_bounds (:obj:`tensor`): The parameter value bounds to be used for fitting.
            fixed_params (:obj:`tensor`): Whether to fix the parameter to the init_pars value during minimization
            test_stat (:obj:`str`): The test statistic to use as a numerical summary of the data.
            qtilde (:obj:`bool`): When ``True`` perform the calculation using the alternative
             test statistic, :math:`\tilde{q}_{\mu}`, as defined under the Wald
             approximation in Equation (62) of :xref:`arXiv:1007.1727`
             (:func:`~pyhf.infer.test_statistics.qmu_tilde`).
             When ``False`` use :func:`~pyhf.infer.test_statistics.qmu`.

        Returns:
            ~pyhf.infer.calculators.AsymptoticCalculator: The calculator for asymptotic quantities.

        """
        self.data = data
        self.pdf = pdf
        self.init_pars = init_pars or pdf.config.suggested_init()
        self.par_bounds = par_bounds or pdf.config.suggested_bounds()
        self.fixed_params = fixed_params or pdf.config.suggested_fixed()
        self.test_stat = test_stat
        self.sqrtqmuA_v = None

    def distributions(self, poi_test):
        r"""
        Probability distributions of the test statistic, as defined in
        :math:`\S` 3 of :xref:`arXiv:1007.1727` under the Wald approximation,
        under the signal + background and background-only hypotheses.

        Example:

            >>> import pyhf
            >>> pyhf.set_backend("numpy")
            >>> model = pyhf.simplemodels.hepdata_like(
            ...     signal_data=[12.0, 11.0], bkg_data=[50.0, 52.0], bkg_uncerts=[3.0, 7.0]
            ... )
            >>> observations = [51, 48]
            >>> data = observations + model.config.auxdata
            >>> mu_test = 1.0
            >>> asymptotic_calculator = pyhf.infer.calculators.AsymptoticCalculator(data, model, test_stat="qtilde")
            >>> _ = asymptotic_calculator.teststatistic(mu_test)
            >>> sig_plus_bkg_dist, bkg_dist = asymptotic_calculator.distributions(mu_test)
            >>> sig_plus_bkg_dist.pvalue(mu_test), bkg_dist.pvalue(mu_test)
            (0.0021926462553507452, 0.15865525393145707)

        Args:
            poi_test (:obj:`float` or :obj:`tensor`): The value for the parameter of interest.

        Returns:
            Tuple (~pyhf.infer.calculators.AsymptoticTestStatDistribution): The distributions under the hypotheses.

        """
        if self.sqrtqmuA_v is None:
            raise RuntimeError('need to call .teststatistic(poi_test) first')
        sb_dist = AsymptoticTestStatDistribution(-self.sqrtqmuA_v)
        b_dist = AsymptoticTestStatDistribution(0.0)
        return sb_dist, b_dist

    def teststatistic(self, poi_test):
        """
        Compute the test statistic for the observed data under the studied model.

        Example:

            >>> import pyhf
            >>> pyhf.set_backend("numpy")
            >>> model = pyhf.simplemodels.hepdata_like(
            ...     signal_data=[12.0, 11.0], bkg_data=[50.0, 52.0], bkg_uncerts=[3.0, 7.0]
            ... )
            >>> observations = [51, 48]
            >>> data = observations + model.config.auxdata
            >>> mu_test = 1.0
            >>> asymptotic_calculator = pyhf.infer.calculators.AsymptoticCalculator(data, model, test_stat="qtilde")
            >>> asymptotic_calculator.teststatistic(mu_test)
            0.1404353407503529

        Args:
            poi_test (:obj:`float` or :obj:`tensor`): The value for the parameter of interest.

        Returns:
            Float: The value of the test statistic.

        """
        tensorlib, _ = get_backend()

        teststat_func = utils.get_test_stat(self.test_stat)

        qmu_v = teststat_func(
            poi_test,
            self.data,
            self.pdf,
            self.init_pars,
            self.par_bounds,
            self.fixed_params,
        )
        sqrtqmu_v = tensorlib.sqrt(qmu_v)

        asimov_mu = 1.0 if self.test_stat == 'q0' else 0.0

        asimov_data = generate_asimov_data(
            asimov_mu,
            self.data,
            self.pdf,
            self.init_pars,
            self.par_bounds,
            self.fixed_params,
        )
        qmuA_v = teststat_func(
            poi_test,
            asimov_data,
            self.pdf,
            self.init_pars,
            self.par_bounds,
            self.fixed_params,
        )
        self.sqrtqmuA_v = tensorlib.sqrt(qmuA_v)

        if self.test_stat in ["q", "q0"]:  # qmu or q0
            teststat = sqrtqmu_v - self.sqrtqmuA_v
        else:  # qtilde

            def _true_case():
                teststat = sqrtqmu_v - self.sqrtqmuA_v
                return teststat

            def _false_case():
                qmu = tensorlib.power(sqrtqmu_v, 2)
                qmu_A = tensorlib.power(self.sqrtqmuA_v, 2)
                teststat = (qmu - qmu_A) / (2 * self.sqrtqmuA_v)
                return teststat

            teststat = tensorlib.conditional(
                (sqrtqmu_v < self.sqrtqmuA_v), _true_case, _false_case
            )
        return teststat

    def pvalues(self, teststat, sig_plus_bkg_distribution, bkg_only_distribution):
        r"""
        Calculate the :math:`p`-values for the observed test statistic under the
        signal + background and background-only model hypotheses.

        Example:

            >>> import pyhf
            >>> pyhf.set_backend("numpy")
            >>> model = pyhf.simplemodels.hepdata_like(
            ...     signal_data=[12.0, 11.0], bkg_data=[50.0, 52.0], bkg_uncerts=[3.0, 7.0]
            ... )
            >>> observations = [51, 48]
            >>> data = observations + model.config.auxdata
            >>> mu_test = 1.0
            >>> asymptotic_calculator = pyhf.infer.calculators.AsymptoticCalculator(
            ...     data, model, test_stat="qtilde"
            ... )
            >>> q_tilde = asymptotic_calculator.teststatistic(mu_test)
            >>> sig_plus_bkg_dist, bkg_dist = asymptotic_calculator.distributions(mu_test)
            >>> CLsb, CLb, CLs = asymptotic_calculator.pvalues(q_tilde, sig_plus_bkg_dist, bkg_dist)
            >>> CLsb, CLb, CLs
            (0.023325003824589668, 0.44415801831909385, 0.05251510242427374)

        Args:
            teststat (:obj:`tensor`): The test statistic.
            sig_plus_bkg_distribution (~pyhf.infer.calculators.AsymptoticTestStatDistribution):
              The distribution for the signal + background hypothesis.
            bkg_only_distribution (~pyhf.infer.calculators.AsymptoticTestStatDistribution):
              The distribution for the background-only hypothesis.

        Returns:
            Tuple (:obj:`float`): The :math:`p`-values for the test statistic
            corresponding to the :math:`\mathrm{CL}_{s+b}`,
            :math:`\mathrm{CL}_{b}`, and :math:`\mathrm{CL}_{s}`.
        """
        CLsb = sig_plus_bkg_distribution.pvalue(teststat)
        CLb = bkg_only_distribution.pvalue(teststat)
        CLs = CLsb / CLb
        return CLsb, CLb, CLs

    def expected_pvalues(self, sig_plus_bkg_distribution, bkg_only_distribution):
        r"""
        Calculate the :math:`\mathrm{CL}_{s}` values corresponding to the
        median significance of variations of the signal strength from the
        background only hypothesis :math:`\left(\mu=0\right)` at
        :math:`(-2,-1,0,1,2)\sigma`.

        Example:

            >>> import pyhf
            >>> pyhf.set_backend("numpy")
            >>> model = pyhf.simplemodels.hepdata_like(
            ...     signal_data=[12.0, 11.0], bkg_data=[50.0, 52.0], bkg_uncerts=[3.0, 7.0]
            ... )
            >>> observations = [51, 48]
            >>> data = observations + model.config.auxdata
            >>> mu_test = 1.0
            >>> asymptotic_calculator = pyhf.infer.calculators.AsymptoticCalculator(
            ...     data, model, test_stat="qtilde"
            ... )
            >>> _ = asymptotic_calculator.teststatistic(mu_test)
            >>> sig_plus_bkg_dist, bkg_dist = asymptotic_calculator.distributions(mu_test)
            >>> CLsb_exp_band, CLb_exp_band, CLs_exp_band = asymptotic_calculator.expected_pvalues(sig_plus_bkg_dist, bkg_dist)
            >>> CLs_exp_band
            [0.002606295139302753, 0.013820193161066205, 0.06445366938143944, 0.23525750131662554, 0.573037502984585]

        Args:
            sig_plus_bkg_distribution (~pyhf.infer.calculators.AsymptoticTestStatDistribution):
              The distribution for the signal + background hypothesis.
            bkg_only_distribution (~pyhf.infer.calculators.AsymptoticTestStatDistribution):
              The distribution for the background-only hypothesis.

        Returns:
            Tuple (:obj:`float`): The :math:`p`-values for the test statistic
            corresponding to the :math:`\mathrm{CL}_{s+b}`,
            :math:`\mathrm{CL}_{b}`, and :math:`\mathrm{CL}_{s}`.
        """
        # Calling pvalues is easier then repeating the CLs calculation here
        tb, _ = get_backend()
        return list(
            map(
                list,
                zip(
                    *[
                        self.pvalues(
                            test_stat, sig_plus_bkg_distribution, bkg_only_distribution
                        )
                        for test_stat in [
                            bkg_only_distribution.expected_value(n_sigma)
                            for n_sigma in [2, 1, 0, -1, -2]
                        ]
                    ]
                ),
            )
        )


class EmpiricalDistribution:
    """
    The empirical distribution of the test statistic.

    Unlike :py:class:`~pyhf.infer.calculators.AsymptoticTestStatDistribution` where the
    distribution for the test statistic is normally distributed, the
    :math:`p`-values etc are computed from the sampled distribution.
    """

    def __init__(self, samples):
        """
        Empirical distribution.

        Args:
            samples (:obj:`tensor`): The test statistics sampled from the distribution.

        Returns:
            ~pyhf.infer.calculators.EmpiricalDistribution: The empirical distribution of the test statistic.

        """
        tensorlib, _ = get_backend()
        self.samples = tensorlib.ravel(samples)

    def pvalue(self, value):
        """
        Compute the :math:`p`-value for a given value of the test statistic.

        Examples:

            >>> import pyhf
            >>> import numpy.random as random
            >>> random.seed(0)
            >>> pyhf.set_backend("numpy")
            >>> mean = pyhf.tensorlib.astensor([5])
            >>> std = pyhf.tensorlib.astensor([1])
            >>> normal = pyhf.probability.Normal(mean, std)
            >>> samples = normal.sample((100,))
            >>> dist = pyhf.infer.calculators.EmpiricalDistribution(samples)
            >>> dist.pvalue(7)
            0.02

            >>> import pyhf
            >>> import numpy.random as random
            >>> random.seed(0)
            >>> pyhf.set_backend("numpy")
            >>> model = pyhf.simplemodels.hepdata_like(
            ...     signal_data=[12.0, 11.0], bkg_data=[50.0, 52.0], bkg_uncerts=[3.0, 7.0]
            ... )
            >>> init_pars = model.config.suggested_init()
            >>> par_bounds = model.config.suggested_bounds()
            >>> fixed_params = model.config.suggested_fixed()
            >>> mu_test = 1.0
            >>> pdf = model.make_pdf(pyhf.tensorlib.astensor(init_pars))
            >>> samples = pdf.sample((100,))
            >>> test_stat_dist = pyhf.infer.calculators.EmpiricalDistribution(
            ...     pyhf.tensorlib.astensor(
            ...         [pyhf.infer.test_statistics.qmu_tilde(mu_test, sample, model, init_pars, par_bounds, fixed_params) for sample in samples]
            ...     )
            ... )
            >>> test_stat_dist.pvalue(test_stat_dist.samples[9])
            0.3

        Args:
            value (:obj:`float`): The test statistic value.

        Returns:
            Float: The integrated probability to observe a value at least as large as the observed one.

        """
        tensorlib, _ = get_backend()
        return (
            tensorlib.sum(
                tensorlib.where(
                    self.samples >= value, tensorlib.astensor(1), tensorlib.astensor(0)
                )
            )
            / tensorlib.shape(self.samples)[0]
        )

    def expected_value(self, nsigma):
        """
        Return the expected value of the test statistic.

        Examples:

            >>> import pyhf
            >>> import numpy.random as random
            >>> random.seed(0)
            >>> pyhf.set_backend("numpy")
            >>> mean = pyhf.tensorlib.astensor([5])
            >>> std = pyhf.tensorlib.astensor([1])
            >>> normal = pyhf.probability.Normal(mean, std)
            >>> samples = normal.sample((100,))
            >>> dist = pyhf.infer.calculators.EmpiricalDistribution(samples)
            >>> dist.expected_value(nsigma=1)
            6.15094381209505

            >>> import pyhf
            >>> import numpy.random as random
            >>> random.seed(0)
            >>> pyhf.set_backend("numpy")
            >>> model = pyhf.simplemodels.hepdata_like(
            ...     signal_data=[12.0, 11.0], bkg_data=[50.0, 52.0], bkg_uncerts=[3.0, 7.0]
            ... )
            >>> init_pars = model.config.suggested_init()
            >>> par_bounds = model.config.suggested_bounds()
            >>> fixed_params = model.config.suggested_fixed()
            >>> mu_test = 1.0
            >>> pdf = model.make_pdf(pyhf.tensorlib.astensor(init_pars))
            >>> samples = pdf.sample((100,))
            >>> dist = pyhf.infer.calculators.EmpiricalDistribution(
            ...     pyhf.tensorlib.astensor(
            ...         [
            ...             pyhf.infer.test_statistics.qmu_tilde(
            ...                 mu_test, sample, model, init_pars, par_bounds, fixed_params
            ...             )
            ...             for sample in samples
            ...         ]
            ...     )
            ... )
            >>> n_sigma = pyhf.tensorlib.astensor([-2, -1, 0, 1, 2])
            >>> dist.expected_value(n_sigma)
            array([0.00000000e+00, 0.00000000e+00, 5.54127113e-04, 8.29987224e-01,
                   2.99592670e+00])

        Args:
            nsigma (:obj:`int` or :obj:`tensor`): The number of standard deviations.

        Returns:
            Float: The expected value of the test statistic.
        """
        tensorlib, _ = get_backend()
        import numpy as np

        # TODO: tensorlib.percentile function
        # c.f. https://github.com/scikit-hep/pyhf/pull/817
        return np.percentile(
            self.samples, tensorlib.normal_cdf(nsigma) * 100, interpolation="linear"
        )


class ToyCalculator:
    """The Toy-based Calculator."""

    def __init__(
        self,
        data,
        pdf,
        init_pars=None,
        par_bounds=None,
        fixed_params=None,
        test_stat="qtilde",
        ntoys=2000,
        track_progress=True,
    ):
        r"""
        Toy-based Calculator.

        Args:
            data (:obj:`tensor`): The observed data.
            pdf (~pyhf.pdf.Model): The statistical model adhering to the schema ``model.json``.
            init_pars (:obj:`tensor`): The initial parameter values to be used for fitting.
            par_bounds (:obj:`tensor`): The parameter value bounds to be used for fitting.
            fixed_params (:obj:`tensor`): Whether to fix the parameter to the init_pars value during minimization
            test_stat (:obj:`str`): The test statistic to use as a numerical summary of the data.
            qtilde (:obj:`bool`): When ``True`` perform the calculation using the alternative
             test statistic, :math:`\tilde{q}_{\mu}`, as defined under the Wald
             approximation in Equation (62) of :xref:`arXiv:1007.1727`
             (:func:`~pyhf.infer.test_statistics.qmu_tilde`).
             When ``False`` use :func:`~pyhf.infer.test_statistics.qmu`.
            ntoys (:obj:`int`): Number of toys to use (how many times to sample the underlying distributions)
            track_progress (:obj:`bool`): Whether to display the `tqdm` progress bar or not (outputs to `stderr`)

        Returns:
            ~pyhf.infer.calculators.ToyCalculator: The calculator for toy-based quantities.

        """
        self.ntoys = ntoys
        self.data = data
        self.pdf = pdf
        self.init_pars = init_pars or pdf.config.suggested_init()
        self.par_bounds = par_bounds or pdf.config.suggested_bounds()
        self.fixed_params = fixed_params or pdf.config.suggested_fixed()
        self.test_stat = test_stat
        self.track_progress = track_progress

    def distributions(self, poi_test, track_progress=None):
        """
        Probability Distributions of the test statistic value under the signal + background and background-only hypothesis.

        Example:

            >>> import pyhf
            >>> import numpy.random as random
            >>> random.seed(0)
            >>> pyhf.set_backend("numpy")
            >>> model = pyhf.simplemodels.hepdata_like(
            ...     signal_data=[12.0, 11.0], bkg_data=[50.0, 52.0], bkg_uncerts=[3.0, 7.0]
            ... )
            >>> observations = [51, 48]
            >>> data = observations + model.config.auxdata
            >>> mu_test = 1.0
            >>> toy_calculator = pyhf.infer.calculators.ToyCalculator(
            ...     data, model, ntoys=100, track_progress=False
            ... )
            >>> sig_plus_bkg_dist, bkg_dist = toy_calculator.distributions(mu_test)
            >>> sig_plus_bkg_dist.pvalue(mu_test), bkg_dist.pvalue(mu_test)
            (0.14, 0.76)

        Args:
            poi_test (:obj:`float` or :obj:`tensor`): The value for the parameter of interest.
            track_progress (:obj:`bool`): Whether to display the `tqdm` progress bar or not (outputs to `stderr`)

        Returns:
            Tuple (~pyhf.infer.calculators.EmpiricalDistribution): The distributions under the hypotheses.

        """
        tensorlib, _ = get_backend()
        sample_shape = (self.ntoys,)

        signal_pars = self.pdf.config.suggested_init()
        signal_pars[self.pdf.config.poi_index] = poi_test
        signal_pdf = self.pdf.make_pdf(tensorlib.astensor(signal_pars))
        signal_sample = signal_pdf.sample(sample_shape)

        bkg_pars = self.pdf.config.suggested_init()
        bkg_pars[self.pdf.config.poi_index] = 1.0 if self.test_stat == 'q0' else 0.0
        bkg_pdf = self.pdf.make_pdf(tensorlib.astensor(bkg_pars))
        bkg_sample = bkg_pdf.sample(sample_shape)

        teststat_func = utils.get_test_stat(self.test_stat)

        tqdm_options = dict(
            total=self.ntoys,
            leave=False,
            disable=not (
                track_progress if track_progress is not None else self.track_progress
            ),
            unit='toy',
        )

        signal_teststat = []
        for sample in tqdm.tqdm(signal_sample, **tqdm_options, desc='Signal-like'):
            signal_teststat.append(
                teststat_func(
                    poi_test,
                    sample,
                    self.pdf,
                    self.init_pars,
                    self.par_bounds,
                    self.fixed_params,
                )
            )

        bkg_teststat = []
        for sample in tqdm.tqdm(bkg_sample, **tqdm_options, desc='Background-like'):
            bkg_teststat.append(
                teststat_func(
                    poi_test,
                    sample,
                    self.pdf,
                    self.init_pars,
                    self.par_bounds,
                    self.fixed_params,
                )
            )

        s_plus_b = EmpiricalDistribution(tensorlib.astensor(signal_teststat))
        b_only = EmpiricalDistribution(tensorlib.astensor(bkg_teststat))
        return s_plus_b, b_only

    def pvalues(self, teststat, sig_plus_bkg_distribution, bkg_only_distribution):
        r"""
        Calculate the :math:`p`-values for the observed test statistic under the
        signal + background and background-only model hypotheses.

        Example:

            >>> import pyhf
            >>> import numpy.random as random
            >>> random.seed(0)
            >>> pyhf.set_backend("numpy")
            >>> model = pyhf.simplemodels.hepdata_like(
            ...     signal_data=[12.0, 11.0], bkg_data=[50.0, 52.0], bkg_uncerts=[3.0, 7.0]
            ... )
            >>> observations = [51, 48]
            >>> data = observations + model.config.auxdata
            >>> mu_test = 1.0
            >>> toy_calculator = pyhf.infer.calculators.ToyCalculator(
            ...     data, model, ntoys=100, track_progress=False
            ... )
            >>> q_tilde = toy_calculator.teststatistic(mu_test)
            >>> sig_plus_bkg_dist, bkg_dist = toy_calculator.distributions(mu_test)
            >>> CLsb, CLb, CLs = toy_calculator.pvalues(q_tilde, sig_plus_bkg_dist, bkg_dist)
            >>> CLsb, CLb, CLs
            (0.01, 0.41, 0.024390243902439025)

        Args:
            teststat (:obj:`tensor`): The test statistic.
            sig_plus_bkg_distribution (~pyhf.infer.calculators.EmpiricalDistribution):
              The distribution for the signal + background hypothesis.
            bkg_only_distribution (~pyhf.infer.calculators.EmpiricalDistribution):
              The distribution for the background-only hypothesis.

        Returns:
            Tuple (:obj:`float`): The :math:`p`-values for the test statistic
            corresponding to the :math:`\mathrm{CL}_{s+b}`,
            :math:`\mathrm{CL}_{b}`, and :math:`\mathrm{CL}_{s}`.
        """
        CLsb = sig_plus_bkg_distribution.pvalue(teststat)
        CLb = bkg_only_distribution.pvalue(teststat)
        CLs = CLsb / CLb
        return CLsb, CLb, CLs

    def expected_pvalues(self, sig_plus_bkg_distribution, bkg_only_distribution):
        r"""
        Calculate the :math:`\mathrm{CL}_{s}` values corresponding to the
        median significance of variations of the signal strength from the
        background only hypothesis :math:`\left(\mu=0\right)` at
        :math:`(-2,-1,0,1,2)\sigma`.

        Example:

            >>> import pyhf
            >>> import numpy.random as random
            >>> random.seed(0)
            >>> pyhf.set_backend("numpy")
            >>> model = pyhf.simplemodels.hepdata_like(
            ...     signal_data=[12.0, 11.0], bkg_data=[50.0, 52.0], bkg_uncerts=[3.0, 7.0]
            ... )
            >>> observations = [51, 48]
            >>> data = observations + model.config.auxdata
            >>> mu_test = 1.0
            >>> toy_calculator = pyhf.infer.calculators.ToyCalculator(
            ...     data, model, ntoys=100, track_progress=False
            ... )
            >>> sig_plus_bkg_dist, bkg_dist = toy_calculator.distributions(mu_test)
            >>> CLsb_exp_band, CLb_exp_band, CLs_exp_band = toy_calculator.expected_pvalues(sig_plus_bkg_dist, bkg_dist)
            >>> CLs_exp_band
            [0.0, 0.0, 0.06186224489795918, 0.2845003327965815, 1.0]

        Args:
            sig_plus_bkg_distribution (~pyhf.infer.calculators.EmpiricalDistribution):
              The distribution for the signal + background hypothesis.
            bkg_only_distribution (~pyhf.infer.calculators.EmpiricalDistribution):
              The distribution for the background-only hypothesis.

        Returns:
            Tuple (:obj:`float`): The :math:`p`-values for the test statistic
            corresponding to the :math:`\mathrm{CL}_{s+b}`,
            :math:`\mathrm{CL}_{b}`, and :math:`\mathrm{CL}_{s}`.
        """
        tb, _ = get_backend()
        pvalues = tb.astensor(
            [
                self.pvalues(
                    tb.astensor(test_stat),
                    sig_plus_bkg_distribution,
                    bkg_only_distribution,
                )
                for test_stat in bkg_only_distribution.samples
            ]
        )
        # TODO: Add percentile to tensorlib
        # c.f. Issue #815, PR #817
        import numpy as np

        # percentiles for -2, -1, 0, 1, 2 standard deviations of the Normal distribution
        normal_percentiles = [2.27501319, 15.86552539, 50.0, 84.13447461, 97.72498681]
        pvalues_exp_band = np.percentile(
            tb.tolist(pvalues),
            normal_percentiles,
            axis=0,
        ).T.tolist()
        return pvalues_exp_band

    def teststatistic(self, poi_test):
        """
        Compute the test statistic for the observed data under the studied model.

        Example:

            >>> import pyhf
            >>> import numpy.random as random
            >>> random.seed(0)
            >>> pyhf.set_backend("numpy")
            >>> model = pyhf.simplemodels.hepdata_like(
            ...     signal_data=[12.0, 11.0], bkg_data=[50.0, 52.0], bkg_uncerts=[3.0, 7.0]
            ... )
            >>> observations = [51, 48]
            >>> data = observations + model.config.auxdata
            >>> mu_test = 1.0
            >>> toy_calculator = pyhf.infer.calculators.ToyCalculator(
            ...     data, model, ntoys=100, track_progress=False
            ... )
            >>> toy_calculator.teststatistic(mu_test)
            array(3.93824506)

        Args:
            poi_test (:obj:`float` or :obj:`tensor`): The value for the parameter of interest.

        Returns:
            Float: The value of the test statistic.

        """
        teststat_func = utils.get_test_stat(self.test_stat)
        teststat = teststat_func(
            poi_test,
            self.data,
            self.pdf,
            self.init_pars,
            self.par_bounds,
            self.fixed_params,
        )
        return teststat
