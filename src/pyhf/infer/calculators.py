"""
Calculators for Hypothesis Testing.

The role of the calculators is to compute test statistic and
provide distributions of said test statistic under various
hypotheses.

Using the calculators hypothesis tests can then be performed.
"""
from pyhf.infer.mle import fixed_poi_fit
from pyhf import get_backend
from pyhf.infer import utils
import tqdm

from dataclasses import dataclass
import logging

log = logging.getLogger(__name__)

__all__ = [
    "AsymptoticCalculator",
    "AsymptoticTestStatDistribution",
    "EmpiricalDistribution",
    "ToyCalculator",
    "generate_asimov_data",
]


def __dir__():
    return __all__


def generate_asimov_data(
    asimov_mu, data, pdf, init_pars, par_bounds, fixed_params, return_fitted_pars=False
):
    """
    Compute Asimov Dataset (expected yields at best-fit values) for a given POI value.

    Example:

        >>> import pyhf
        >>> pyhf.set_backend("numpy")
        >>> model = pyhf.simplemodels.uncorrelated_background(
        ...     signal=[12.0, 11.0], bkg=[50.0, 52.0], bkg_uncertainty=[3.0, 7.0]
        ... )
        >>> observations = [51, 48]
        >>> data = observations + model.config.auxdata
        >>> mu_test = 1.0
        >>> pyhf.infer.calculators.generate_asimov_data(mu_test, data, model, None, None, None)
        array([ 60.61229858,  56.52802479, 270.06832542,  48.31545488])
        >>> pyhf.infer.calculators.generate_asimov_data(
        ...     mu_test, data, model, None, None, None, return_fitted_pars=True
        ... )
        (array([ 60.61229858,  56.52802479, 270.06832542,  48.31545488]), array([1.        , 0.97224597, 0.87553894]))

    Args:
        asimov_mu (:obj:`float`): The value for the parameter of interest to be used.
        data (:obj:`tensor`): The observed data.
        pdf (~pyhf.pdf.Model): The statistical model adhering to the schema ``model.json``.
        init_pars (:obj:`tensor` of :obj:`float`): The starting values of the model parameters for minimization.
        par_bounds (:obj:`tensor`): The extrema of values the model parameters
            are allowed to reach in the fit.
            The shape should be ``(n, 2)`` for ``n`` model parameters.
        fixed_params (:obj:`tensor` of :obj:`bool`): The flag to set a parameter constant to its starting
            value during minimization.
        return_fitted_pars (:obj:`bool`): Return the best-fit parameter values for the given ``asimov_mu``.


    Returns:
        A Tensor or a Tuple of two Tensors:

             - The Asimov dataset.

             - The Asimov parameters. Only returned if ``return_fitted_pars`` is ``True``.
    """
    bestfit_nuisance_asimov = fixed_poi_fit(
        asimov_mu, data, pdf, init_pars, par_bounds, fixed_params
    )
    asimov_data = pdf.expected_data(bestfit_nuisance_asimov)
    if return_fitted_pars:
        return asimov_data, bestfit_nuisance_asimov
    return asimov_data


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

    def __init__(self, shift, cutoff=float("-inf")):
        """
        Asymptotic test statistic distribution.

        Args:
            shift (:obj:`float`): The displacement of the test statistic distribution.

        Returns:
            ~pyhf.infer.calculators.AsymptoticTestStatDistribution: The asymptotic distribution of test statistic.

        """
        self.shift = shift
        self.cutoff = cutoff

    def cdf(self, value):
        """
        Compute the value of the cumulative distribution function for a given value of the test statistic.

        Example:

            >>> import pyhf
            >>> pyhf.set_backend("numpy")
            >>> bkg_dist = pyhf.infer.calculators.AsymptoticTestStatDistribution(0.0)
            >>> bkg_dist.cdf(0.0)
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

        Example:

            >>> import pyhf
            >>> pyhf.set_backend("numpy")
            >>> bkg_dist = pyhf.infer.calculators.AsymptoticTestStatDistribution(0.0)
            >>> bkg_dist.pvalue(0.0)
            array(0.5)

        Args:
            value (:obj:`float`): The test statistic value.

        Returns:
            Tensor: The integrated probability to observe a value at least as large as the observed one.

        """
        tensorlib, _ = get_backend()
        # computing cdf(-x) instead of 1-cdf(x) for right-tail p-value for improved numerical stability

        return_value = tensorlib.normal_cdf(-(value - self.shift))
        invalid_value = tensorlib.ones(tensorlib.shape(return_value)) * float("nan")
        return tensorlib.where(
            tensorlib.astensor(value >= self.cutoff, dtype="bool"),
            return_value,
            invalid_value,
        )

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
        tensorlib, _ = get_backend()
        return tensorlib.where(
            tensorlib.astensor(self.shift + nsigma > self.cutoff, dtype="bool"),
            tensorlib.astensor(self.shift + nsigma),
            tensorlib.astensor(self.cutoff),
        )


@dataclass(frozen=True)
class HypoTestFitResults:
    """
    Fitted model parameters of the fits in
    :py:meth:`AsymptoticCalculator.teststatistic <pyhf.infer.calculators.AsymptoticCalculator.teststatistic>`
    """

    # ignore "F821 undefined name 'Tensor'" so as to avoid typing.Any
    asimov_pars: 'Tensor'  # noqa: F821
    free_fit_to_data: 'Tensor'  # noqa: F821
    free_fit_to_asimov: 'Tensor'  # noqa: F821
    fixed_poi_fit_to_data: 'Tensor'  # noqa: F821
    fixed_poi_fit_to_asimov: 'Tensor'  # noqa: F821


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
        calc_base_dist="normal",
    ):
        r"""
        Asymptotic Calculator.

        Args:
            data (:obj:`tensor`): The observed data.
            pdf (~pyhf.pdf.Model): The statistical model adhering to the schema ``model.json``.
            init_pars (:obj:`tensor` of :obj:`float`): The starting values of the model parameters for minimization.
            par_bounds (:obj:`tensor`): The extrema of values the model parameters
                are allowed to reach in the fit.
                The shape should be ``(n, 2)`` for ``n`` model parameters.
            fixed_params (:obj:`tensor` of :obj:`bool`): The flag to set a parameter constant to its starting
                value during minimization.
            test_stat (:obj:`str`): The test statistic to use as a numerical summary of the
              data: ``'qtilde'``, ``'q'``, or ``'q0'``.

              * ``'qtilde'``: (default) performs the calculation using the alternative test statistic,
                :math:`\tilde{q}_{\mu}`, as defined under the Wald approximation in Equation (62)
                of :xref:`arXiv:1007.1727` (:func:`~pyhf.infer.test_statistics.qmu_tilde`).
              * ``'q'``: performs the calculation using the test statistic :math:`q_{\mu}`
                (:func:`~pyhf.infer.test_statistics.qmu`).
              * ``'q0'``: performs the calculation using the discovery test statistic
                :math:`q_{0}` (:func:`~pyhf.infer.test_statistics.q0`).
            calc_base_dist (:obj:`str`): The statistical distribution, ``'normal'`` or
              ``'clipped_normal'``, to use for calculating the :math:`p`-values.

              * ``'normal'``: (default) use the full Normal distribution in :math:`\hat{\mu}/\sigma`
                space.
                Note that expected limits may correspond to unphysical test statistics from scenarios
                with the expected :math:`\hat{\mu} > \mu`.
              * ``'clipped_normal'``: use a clipped Normal distribution in :math:`\hat{\mu}/\sigma`
                space to avoid expected limits that correspond to scenarios with the expected
                :math:`\hat{\mu} > \mu`.
                This will properly cap the test statistic at ``0``, as noted in Equation (14) and
                Equation (16) in :xref:`arXiv:1007.1727`.

              The choice of ``calc_base_dist`` only affects the :math:`p`-values for expected limits,
              and the default value will be changed in a future release.

        Returns:
            ~pyhf.infer.calculators.AsymptoticCalculator: The calculator for asymptotic quantities.

        """
        self.data = data
        self.pdf = pdf
        self.init_pars = init_pars or pdf.config.suggested_init()
        self.par_bounds = par_bounds or pdf.config.suggested_bounds()
        self.fixed_params = fixed_params or pdf.config.suggested_fixed()
        self.test_stat = test_stat
        self.calc_base_dist = calc_base_dist
        self.sqrtqmuA_v = None
        self.fitted_pars = None

    def distributions(self, poi_test):
        r"""
        Probability distributions of the test statistic, as defined in
        :math:`\S` 3 of :xref:`arXiv:1007.1727` under the Wald approximation,
        under the signal + background and background-only hypotheses.

        Example:

            >>> import pyhf
            >>> pyhf.set_backend("numpy")
            >>> model = pyhf.simplemodels.uncorrelated_background(
            ...     signal=[12.0, 11.0], bkg=[50.0, 52.0], bkg_uncertainty=[3.0, 7.0]
            ... )
            >>> observations = [51, 48]
            >>> data = observations + model.config.auxdata
            >>> mu_test = 1.0
            >>> asymptotic_calculator = pyhf.infer.calculators.AsymptoticCalculator(data, model, test_stat="qtilde")
            >>> _ = asymptotic_calculator.teststatistic(mu_test)
            >>> sig_plus_bkg_dist, bkg_dist = asymptotic_calculator.distributions(mu_test)
            >>> sig_plus_bkg_dist.pvalue(mu_test), bkg_dist.pvalue(mu_test)
            (array(0.00219262), array(0.15865525))

        Args:
            poi_test (:obj:`float` or :obj:`tensor`): The value for the parameter of interest.

        Returns:
            Tuple (~pyhf.infer.calculators.AsymptoticTestStatDistribution): The distributions under the hypotheses.

        """
        if self.sqrtqmuA_v is None:
            raise RuntimeError("need to call .teststatistic(poi_test) first")

        if self.calc_base_dist == "normal":
            cutoff = float("-inf")
        elif self.calc_base_dist == "clipped_normal":
            cutoff = -self.sqrtqmuA_v
        else:
            raise ValueError(
                f"unknown base distribution for asymptotics {self.calc_base_dist}"
            )
        sb_dist = AsymptoticTestStatDistribution(-self.sqrtqmuA_v, cutoff)
        b_dist = AsymptoticTestStatDistribution(0.0, cutoff)
        return sb_dist, b_dist

    def teststatistic(self, poi_test):
        r"""
        Compute the test statistic for the observed data under the studied model.

        The fitted parameters of the five fits that are implicitly run for each call
        of this method are afterwards accessible through the ``fitted_pars`` attribute,
        which is a :py:class:`~pyhf.infer.calculators.HypoTestFitResults` instance.

        Example:

            >>> import pyhf
            >>> pyhf.set_backend("numpy")
            >>> model = pyhf.simplemodels.uncorrelated_background(
            ...     signal=[12.0, 11.0], bkg=[50.0, 52.0], bkg_uncertainty=[3.0, 7.0]
            ... )
            >>> observations = [51, 48]
            >>> data = observations + model.config.auxdata
            >>> mu_test = 1.0
            >>> asymptotic_calculator = pyhf.infer.calculators.AsymptoticCalculator(data, model, test_stat="qtilde")
            >>> asymptotic_calculator.teststatistic(mu_test)
            array(0.14043184)
            >>> asymptotic_calculator.fitted_pars
            HypoTestFitResults(asimov_pars=array([0.        , 1.0030482 , 0.96264534]), free_fit_to_data=array([0.        , 1.0030512 , 0.96266961]), free_fit_to_asimov=array([0.        , 1.00304893, 0.96263365]), fixed_poi_fit_to_data=array([1.        , 0.97224597, 0.87553894]), fixed_poi_fit_to_asimov=array([1.        , 0.97276864, 0.87142047]))
            >>> asymptotic_calculator.fitted_pars.free_fit_to_asimov  # best-fit parameters to Asimov dataset
            array([0.        , 1.00304893, 0.96263365])

        Args:
            poi_test (:obj:`float` or :obj:`tensor`): The value for the parameter of interest.

        Returns:
            Tensor: The value of the test statistic.

        """
        tensorlib, _ = get_backend()

        teststat_func = utils.get_test_stat(self.test_stat)

        qmu_v, (mubhathat, muhatbhat) = teststat_func(
            poi_test,
            self.data,
            self.pdf,
            self.init_pars,
            self.par_bounds,
            self.fixed_params,
            return_fitted_pars=True,
        )
        sqrtqmu_v = tensorlib.sqrt(qmu_v)

        asimov_mu = 1.0 if self.test_stat == 'q0' else 0.0

        asimov_data, asimov_mubhathat = generate_asimov_data(
            asimov_mu,
            self.data,
            self.pdf,
            self.init_pars,
            self.par_bounds,
            self.fixed_params,
            return_fitted_pars=True,
        )
        qmuA_v, (mubhathat_A, muhatbhat_A) = teststat_func(
            poi_test,
            asimov_data,
            self.pdf,
            self.init_pars,
            self.par_bounds,
            self.fixed_params,
            return_fitted_pars=True,
        )
        self.sqrtqmuA_v = tensorlib.sqrt(qmuA_v)
        self.fitted_pars = HypoTestFitResults(
            asimov_pars=asimov_mubhathat,
            free_fit_to_data=muhatbhat,
            free_fit_to_asimov=muhatbhat_A,
            fixed_poi_fit_to_data=mubhathat,
            fixed_poi_fit_to_asimov=mubhathat_A,
        )

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
        return tensorlib.astensor(teststat)

    def pvalues(self, teststat, sig_plus_bkg_distribution, bkg_only_distribution):
        r"""
        Calculate the :math:`p`-values for the observed test statistic under the
        signal + background and background-only model hypotheses.

        Example:

            >>> import pyhf
            >>> pyhf.set_backend("numpy")
            >>> model = pyhf.simplemodels.uncorrelated_background(
            ...     signal=[12.0, 11.0], bkg=[50.0, 52.0], bkg_uncertainty=[3.0, 7.0]
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
            (array(0.02332502), array(0.4441594), array(0.05251497))

        Args:
            teststat (:obj:`tensor`): The test statistic.
            sig_plus_bkg_distribution (~pyhf.infer.calculators.AsymptoticTestStatDistribution):
              The distribution for the signal + background hypothesis.
            bkg_only_distribution (~pyhf.infer.calculators.AsymptoticTestStatDistribution):
              The distribution for the background-only hypothesis.

        Returns:
            Tuple (:obj:`tensor`): The :math:`p`-values for the test statistic
            corresponding to the :math:`\mathrm{CL}_{s+b}`,
            :math:`\mathrm{CL}_{b}`, and :math:`\mathrm{CL}_{s}`.
        """
        tensorlib, _ = get_backend()

        CLsb = sig_plus_bkg_distribution.pvalue(teststat)
        CLb = bkg_only_distribution.pvalue(teststat)
        CLs = tensorlib.astensor(CLsb / CLb)
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
            >>> model = pyhf.simplemodels.uncorrelated_background(
            ...     signal=[12.0, 11.0], bkg=[50.0, 52.0], bkg_uncertainty=[3.0, 7.0]
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
            [array(0.00260626), array(0.01382005), array(0.06445321), array(0.23525644), array(0.57303621)]

        Args:
            sig_plus_bkg_distribution (~pyhf.infer.calculators.AsymptoticTestStatDistribution):
              The distribution for the signal + background hypothesis.
            bkg_only_distribution (~pyhf.infer.calculators.AsymptoticTestStatDistribution):
              The distribution for the background-only hypothesis.

        Returns:
            Tuple (:obj:`tensor`): The :math:`p`-values for the test statistic
            corresponding to the :math:`\mathrm{CL}_{s+b}`,
            :math:`\mathrm{CL}_{b}`, and :math:`\mathrm{CL}_{s}`.
        """
        # Calling pvalues is easier then repeating the CLs calculation here
        tb, _ = get_backend()
        return list(
            map(
                list,
                zip(
                    *(
                        self.pvalues(
                            test_stat, sig_plus_bkg_distribution, bkg_only_distribution
                        )
                        for test_stat in [
                            bkg_only_distribution.expected_value(n_sigma)
                            for n_sigma in [2, 1, 0, -1, -2]
                        ]
                    )
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
            array(0.02)

            >>> import pyhf
            >>> import numpy.random as random
            >>> random.seed(0)
            >>> pyhf.set_backend("numpy")
            >>> model = pyhf.simplemodels.uncorrelated_background(
            ...     signal=[12.0, 11.0], bkg=[50.0, 52.0], bkg_uncertainty=[3.0, 7.0]
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
            array(0.3)

        Args:
            value (:obj:`float`): The test statistic value.

        Returns:
            Tensor: The integrated probability to observe a value at least as large as the observed one.

        """
        tensorlib, _ = get_backend()
        return tensorlib.astensor(
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
            6.15094381...

            >>> import pyhf
            >>> import numpy.random as random
            >>> random.seed(0)
            >>> pyhf.set_backend("numpy")
            >>> model = pyhf.simplemodels.uncorrelated_background(
            ...     signal=[12.0, 11.0], bkg=[50.0, 52.0], bkg_uncertainty=[3.0, 7.0]
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
            array([0.00000000e+00, 0.00000000e+00, 5.53671231e-04, 8.29987137e-01,
                   2.99592664e+00])

        Args:
            nsigma (:obj:`int` or :obj:`tensor`): The number of standard deviations.

        Returns:
            Float: The expected value of the test statistic.
        """
        tensorlib, _ = get_backend()
        return tensorlib.percentile(
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
            init_pars (:obj:`tensor` of :obj:`float`): The starting values of the model parameters for minimization.
            par_bounds (:obj:`tensor`): The extrema of values the model parameters
                are allowed to reach in the fit.
                The shape should be ``(n, 2)`` for ``n`` model parameters.
            fixed_params (:obj:`tensor` of :obj:`bool`): The flag to set a parameter constant to its starting
                value during minimization.
            test_stat (:obj:`str`): The test statistic to use as a numerical summary of the
              data: ``'qtilde'``, ``'q'``, or ``'q0'``.
              ``'qtilde'`` (default) performs the calculation using the alternative test statistic,
              :math:`\tilde{q}_{\mu}`, as defined under the Wald approximation in Equation (62)
              of :xref:`arXiv:1007.1727` (:func:`~pyhf.infer.test_statistics.qmu_tilde`), ``'q'``
              performs the calculation using the test statistic :math:`q_{\mu}`
              (:func:`~pyhf.infer.test_statistics.qmu`), and ``'q0'`` performs the calculation using
              the discovery test statistic :math:`q_{0}` (:func:`~pyhf.infer.test_statistics.q0`).
            ntoys (:obj:`int`): Number of toys to use (how many times to sample the underlying distributions).
            track_progress (:obj:`bool`): Whether to display the `tqdm` progress bar or not (outputs to `stderr`).

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
        Probability distributions of the test statistic value under the signal + background and background-only hypotheses.

        These distributions are produced by generating pseudo-data ("toys")
        with the nuisance parameters set to their conditional maximum likelihood
        estimators at the corresponding value of the parameter of interest for
        each hypothesis, following the joint recommendations of the ATLAS and CMS
        experiments in |LHC Higgs search combination procedure|_.

        .. _LHC Higgs search combination procedure: https://inspirehep.net/literature/1196797
        .. |LHC Higgs search combination procedure| replace:: *Procedure for the LHC Higgs boson search combination in Summer 2011*

        Example:

            >>> import pyhf
            >>> import numpy.random as random
            >>> random.seed(0)
            >>> pyhf.set_backend("numpy")
            >>> model = pyhf.simplemodels.uncorrelated_background(
            ...     signal=[12.0, 11.0], bkg=[50.0, 52.0], bkg_uncertainty=[3.0, 7.0]
            ... )
            >>> observations = [51, 48]
            >>> data = observations + model.config.auxdata
            >>> mu_test = 1.0
            >>> toy_calculator = pyhf.infer.calculators.ToyCalculator(
            ...     data, model, ntoys=100, track_progress=False
            ... )
            >>> sig_plus_bkg_dist, bkg_dist = toy_calculator.distributions(mu_test)
            >>> sig_plus_bkg_dist.pvalue(mu_test), bkg_dist.pvalue(mu_test)
            (array(0.14), array(0.79))

        Args:
            poi_test (:obj:`float` or :obj:`tensor`): The value for the parameter of interest.
            track_progress (:obj:`bool`): Whether to display the `tqdm` progress bar or not (outputs to `stderr`)

        Returns:
            Tuple (~pyhf.infer.calculators.EmpiricalDistribution): The distributions under the hypotheses.

        """
        tensorlib, _ = get_backend()
        sample_shape = (self.ntoys,)

        signal_pars = fixed_poi_fit(
            poi_test,
            self.data,
            self.pdf,
            self.init_pars,
            self.par_bounds,
            self.fixed_params,
        )
        signal_pdf = self.pdf.make_pdf(signal_pars)
        signal_sample = signal_pdf.sample(sample_shape)

        bkg_pars = fixed_poi_fit(
            1.0 if self.test_stat == 'q0' else 0.0,
            self.data,
            self.pdf,
            self.init_pars,
            self.par_bounds,
            self.fixed_params,
        )
        bkg_pdf = self.pdf.make_pdf(bkg_pars)
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
            >>> model = pyhf.simplemodels.uncorrelated_background(
            ...     signal=[12.0, 11.0], bkg=[50.0, 52.0], bkg_uncertainty=[3.0, 7.0]
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
            (array(0.03), array(0.37), array(0.08108108))

        Args:
            teststat (:obj:`tensor`): The test statistic.
            sig_plus_bkg_distribution (~pyhf.infer.calculators.EmpiricalDistribution):
              The distribution for the signal + background hypothesis.
            bkg_only_distribution (~pyhf.infer.calculators.EmpiricalDistribution):
              The distribution for the background-only hypothesis.

        Returns:
            Tuple (:obj:`tensor`): The :math:`p`-values for the test statistic
            corresponding to the :math:`\mathrm{CL}_{s+b}`,
            :math:`\mathrm{CL}_{b}`, and :math:`\mathrm{CL}_{s}`.
        """
        tensorlib, _ = get_backend()

        CLsb = sig_plus_bkg_distribution.pvalue(teststat)
        CLb = bkg_only_distribution.pvalue(teststat)
        CLs = tensorlib.astensor(CLsb / CLb)
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
            >>> model = pyhf.simplemodels.uncorrelated_background(
            ...     signal=[12.0, 11.0], bkg=[50.0, 52.0], bkg_uncertainty=[3.0, 7.0]
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
            [array(0.), array(0.), array(0.08403955), array(0.21892596), array(0.86072977)]

        Args:
            sig_plus_bkg_distribution (~pyhf.infer.calculators.EmpiricalDistribution):
              The distribution for the signal + background hypothesis.
            bkg_only_distribution (~pyhf.infer.calculators.EmpiricalDistribution):
              The distribution for the background-only hypothesis.

        Returns:
            Tuple (:obj:`tensor`): The :math:`p`-values for the test statistic
            corresponding to the :math:`\mathrm{CL}_{s+b}`,
            :math:`\mathrm{CL}_{b}`, and :math:`\mathrm{CL}_{s}`.
        """
        tb, _ = get_backend()
        pvalues = tb.astensor(
            [
                self.pvalues(
                    test_stat, sig_plus_bkg_distribution, bkg_only_distribution
                )
                for test_stat in bkg_only_distribution.samples
            ]
        )

        # percentiles for -2, -1, 0, 1, 2 standard deviations of the Normal distribution
        normal_percentiles = tb.astensor(
            [2.27501319, 15.86552539, 50.0, 84.13447461, 97.72498681]
        )

        pvalues_exp_band = tb.transpose(
            tb.percentile(pvalues, normal_percentiles, axis=0)
        )
        return [[tb.astensor(pvalue) for pvalue in band] for band in pvalues_exp_band]

    def teststatistic(self, poi_test):
        """
        Compute the test statistic for the observed data under the studied model.

        Example:

            >>> import pyhf
            >>> import numpy.random as random
            >>> random.seed(0)
            >>> pyhf.set_backend("numpy")
            >>> model = pyhf.simplemodels.uncorrelated_background(
            ...     signal=[12.0, 11.0], bkg=[50.0, 52.0], bkg_uncertainty=[3.0, 7.0]
            ... )
            >>> observations = [51, 48]
            >>> data = observations + model.config.auxdata
            >>> mu_test = 1.0
            >>> toy_calculator = pyhf.infer.calculators.ToyCalculator(
            ...     data, model, ntoys=100, track_progress=False
            ... )
            >>> toy_calculator.teststatistic(mu_test)
            array(3.93824492)

        Args:
            poi_test (:obj:`float` or :obj:`tensor`): The value for the parameter of interest.

        Returns:
            Tensor: The value of the test statistic.

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
