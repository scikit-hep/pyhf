"""
Calculators for Hypothesis Testing.

The role of the calculators is to compute test statistic and
provide distributions of said test statistic under various
hypotheses.

Using the calculators hypothesis tests can then be performed.
"""
from .mle import fixed_poi_fit
from .. import get_backend
from .test_statistics import qmu


def generate_asimov_data(asimov_mu, data, pdf, init_pars, par_bounds):
    """Compute Asimov Dataset (expected yields at best-fit values) for a given POI value."""
    bestfit_nuisance_asimov = fixed_poi_fit(asimov_mu, data, pdf, init_pars, par_bounds)
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
            shift: the displacement of the test statistic distribution

        Returns:
            distribution

        """
        self.shift = shift

    def pvalue(self, value):
        """
        Compute the p-value for a given value of the test statistic.

        Args:
            value: the test statistic value.

        Returns:
            pvalue (float): the integrated probability to observe
            a value at least as large as the observed one.

        """
        tensorlib, _ = get_backend()
        return 1 - tensorlib.normal_cdf(value - self.shift)

    def expected_value(self, nsigma):
        """
        Return the expected value of the test statistic.

        Args:
            nsigma: number of standard deviations.

        Returns:
            expected value (float): the expected value of the test statistic.

        """
        return nsigma


class AsymptoticCalculator(object):
    """The Asymptotic Calculator."""

    def __init__(self, data, pdf, init_pars=None, par_bounds=None, qtilde=False):
        """
        Asymptotic Calculator.

        Args:
            data: the observed data
            pdf: the statistical model
            init_pars: the initial parameters to be used for fitting
            par_bounds: the parameter bounds used for fitting

        Returns:
            calculator

        """
        self.data = data
        self.pdf = pdf
        self.init_pars = init_pars or pdf.config.suggested_init()
        self.par_bounds = par_bounds or pdf.config.suggested_bounds()
        self.qtilde = qtilde

    def distributions(self, poi_test):
        """
        Probability Distributions of the test statistic value under the signal + background and and background-only hypothesis.

        Args:
            poi_test: the value for the parameter of interest.

        Returns
            distributions (Tuple of ~pyhf.infer.calculators.AsymptoticTestStatDistribution): the distributions under the hypotheses.

        """
        sb_dist = AsymptoticTestStatDistribution(-self.sqrtqmuA_v)
        b_dist = AsymptoticTestStatDistribution(0.0)
        return sb_dist, b_dist

    def teststatistic(self, poi_test):
        """
        Compute the test statistic for the observed data under the studied model.

        Args:
            poi_test: the value for the parameter of interest.

        Returns:
            test statistic (Float): the value of the test statistic.

        """
        tensorlib, _ = get_backend()
        qmu_v = qmu(poi_test, self.data, self.pdf, self.init_pars, self.par_bounds)
        sqrtqmu_v = tensorlib.sqrt(qmu_v)

        asimov_mu = 0.0
        asimov_data = generate_asimov_data(
            asimov_mu, self.data, self.pdf, self.init_pars, self.par_bounds
        )
        qmuA_v = qmu(poi_test, asimov_data, self.pdf, self.init_pars, self.par_bounds)
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
