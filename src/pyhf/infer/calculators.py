"""
Calculators for Hypothesis Testing.

The role of the calculators is to compute test statistic and
provide distributions of said test statistic under various
hypotheses.

Using the calculators hypothesis tests can then be performed.
"""
from .mle import fixed_poi_fit
from .. import get_backend
from .test_statistics import qmu, qmu_tilde


def generate_asimov_data(asimov_mu, data, pdf, init_pars, par_bounds, fixed_params):
    """
    Compute Asimov Dataset (expected yields at best-fit values) for a given POI value.

    Args:
        asimov_mu (`float`): The value for the parameter of interest to be used.
        data (`tensor`): The observed data.
        pdf (~pyhf.pdf.Model): The statistical model adhering to the schema ``model.json``.
        init_pars (`tensor`): The initial parameter values to be used for fitting.
        par_bounds (`tensor`): The parameter value bounds to be used for fitting.
        fixed_params (`tensor`): Parameters to be held constant in the fit.

    Returns:
        Tensor: The Asimov dataset.

    """
    bestfit_nuisance_asimov = fixed_poi_fit(
        asimov_mu, data, pdf, init_pars, par_bounds, fixed_params
    )
    return pdf.expected_data(bestfit_nuisance_asimov)


class AsymptoticTestStatDistribution(object):
    """
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
            shift (`float`): The displacement of the test statistic distribution.

        Returns:
            ~pyhf.infer.calculators.AsymptoticTestStatDistribution: The asymptotic distribution of test statistic.

        """
        self.shift = shift
        self.sqrtqmuA_v = None

    def cdf(self, value):
        """
        Compute the value of the cumulative distribution function for a given value of the test statistic.

        Args:
            value (`float`): The test statistic value.

        Returns:
            Float: The integrated probability to observe a test statistic less than or equal to the observed ``value``.

        """
        tensorlib, _ = get_backend()
        return tensorlib.normal_cdf((value - self.shift))

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
            value (`float`): The test statistic value.

        Returns:
            Float: The integrated probability to observe a value at least as large as the observed one.

        """
        tensorlib, _ = get_backend()
        # computing cdf(-x) instead of 1-cdf(x) for right-tail p-value for improved numerical stability
        return tensorlib.normal_cdf(-(value - self.shift))

    def expected_value(self, nsigma):
        """
        Return the expected value of the test statistic.

        Args:
            nsigma (`int` or `tensor`): The number of standard deviations.

        Returns:
            Float: The expected value of the test statistic.
        """
        return self.shift + nsigma


class AsymptoticCalculator(object):
    """The Asymptotic Calculator."""

    def __init__(
        self,
        data,
        pdf,
        init_pars=None,
        par_bounds=None,
        fixed_params=None,
        qtilde=False,
    ):
        """
        Asymptotic Calculator.

        Args:
            data (`tensor`): The observed data.
            pdf (~pyhf.pdf.Model): The statistical model adhering to the schema ``model.json``.
            init_pars (`tensor`): The initial parameter values to be used for fitting.
            par_bounds (`tensor`): The parameter value bounds to be used for fitting.
            fixed_params (`tensor`): Whether to fix the parameter to the init_pars value during minimization
            qtilde (`bool`): Whether to use qtilde as the test statistic.

        Returns:
            ~pyhf.infer.calculators.AsymptoticCalculator: The calculator for asymptotic quantities.

        """
        self.data = data
        self.pdf = pdf
        self.init_pars = init_pars or pdf.config.suggested_init()
        self.par_bounds = par_bounds or pdf.config.suggested_bounds()
        self.fixed_params = fixed_params or pdf.config.suggested_fixed()

        self.qtilde = qtilde
        self.sqrtqmuA_v = None

    def distributions(self, poi_test):
        """
        Probability Distributions of the test statistic value under the signal + background and and background-only hypothesis.

        Args:
            poi_test: The value for the parameter of interest.

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

        Args:
            poi_test: The value for the parameter of interest.

        Returns:
            Float: the value of the test statistic.

        """
        tensorlib, _ = get_backend()

        teststat_func = qmu_tilde if self.qtilde else qmu

        qmu_v = teststat_func(
            poi_test,
            self.data,
            self.pdf,
            self.init_pars,
            self.par_bounds,
            self.fixed_params,
        )
        sqrtqmu_v = tensorlib.sqrt(qmu_v)

        asimov_mu = 0.0
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

        if not self.qtilde:  # qmu
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
