"""Utility Functions for model inference."""
from .. import get_backend
from .mle import fixed_poi_fit
from .test_statistics import qmu


def pvals_from_teststat(
    sqrtqmu_v,
    poi_test, data, pdf, init_pars, par_bounds, 
    qtilde=False):
    r"""
    Compute p-values from test-statistic values.

    The :math:`p`-values for signal strength :math:`\mu` and Asimov strength :math:`\mu'` as defined in Equations (59) and (57) of :xref:`arXiv:1007.1727`

    .. math::

        p_{\mu} = 1-F\left(q_{\mu}\middle|\mu'\right) = 1- \Phi\left(q_{\mu} - \frac{\left(\mu-\mu'\right)}{\sigma}\right)

    with Equation (29)

    .. math::

        \frac{(\mu-\mu')}{\sigma} = \sqrt{\Lambda}= \sqrt{q_{\mu,A}}

    given the observed test statistics :math:`q_{\mu}` and :math:`q_{\mu,A}`.

    Args:
        sqrtqmu_v (Number or Tensor): The root of the calculated test statistic, :math:`\sqrt{q_{\mu}}`
        sqrtqmuA_v (Number or Tensor): The root of the calculated test statistic given the Asimov data, :math:`\sqrt{q_{\mu,A}}`
        qtilde (Bool): When ``True`` perform the calculation using the alternative test statistic, :math:`\tilde{q}`, as defined in Equation (62) of :xref:`arXiv:1007.1727`

    Returns:
        Tuple of Floats: The :math:`p`-values for the signal + background, background only, and signal only hypotheses respectivley

    """
    raise RuntimeError()

from .calculators import AsymptoticTestStatDistribution, AsymptoticCalculator
def pvals_from_teststat_expected(sqrtqmuA_v, nsigma=0):
    r"""
    Compute the expected :math:`p`-values CLsb, CLb and CLs for data corresponding to a given percentile of the alternate hypothesis.

    Args:
        sqrtqmuA_v (Number or Tensor): The root of the calculated test statistic given the Asimov data, :math:`\sqrt{q_{\mu,A}}`
        nsigma (Number or Tensor): The number of standard deviations of variations of the signal strength from the background only hypothesis :math:`\left(\mu=0\right)`

    Returns:
        Tuple of Floats: The :math:`p`-values for the signal + background, background only, and signal only hypotheses respectivley

    """
    # NOTE:
    # To compute the expected p-value, one would need to first compute a hypothetical
    # observed test-statistic for a dataset whose best-fit value is mu^ = mu'-n*sigma:
    # $q_n$, and the proceed with the normal p-value computation for whatever test-statistic
    # was used. However, we can make a shortcut by just computing the p-values in mu^/sigma
    # space, where the p-values are Clsb = cdf(x-sqrt(lambda)) and CLb=cdf(x)

    raise RuntimeError()
