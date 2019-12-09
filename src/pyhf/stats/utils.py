from .. import get_backend


def loglambdav(pars, data, pdf):
    return -2 * pdf.logpdf(pars, data)


def generate_asimov_data(asimov_mu, data, pdf, init_pars, par_bounds):
    _, optimizer = get_backend()
    bestfit_nuisance_asimov = optimizer.constrained_bestfit(
        loglambdav, asimov_mu, data, pdf, init_pars, par_bounds
    )
    return pdf.expected_data(bestfit_nuisance_asimov)


def pvals_from_teststat(sqrtqmu_v, sqrtqmuA_v, qtilde=False):
    r"""
    The :math:`p`-values for signal strength :math:`\mu` and Asimov strength :math:`\mu'` as defined in Equations (59) and (57) of `arXiv:1007.1727`_

    .. _`arXiv:1007.1727`: https://arxiv.org/abs/1007.1727

    .. math::

        p_{\mu} = 1-F\left(q_{\mu}\middle|\mu'\right) = 1- \Phi\left(q_{\mu} - \frac{\left(\mu-\mu'\right)}{\sigma}\right)

    with Equation (29)

    .. math::

        \frac{(\mu-\mu')}{\sigma} = \sqrt{\Lambda}= \sqrt{q_{\mu,A}}

    given the observed test statistics :math:`q_{\mu}` and :math:`q_{\mu,A}`.

    Args:
        sqrtqmu_v (Number or Tensor): The root of the calculated test statistic, :math:`\sqrt{q_{\mu}}`
        sqrtqmuA_v (Number or Tensor): The root of the calculated test statistic given the Asimov data, :math:`\sqrt{q_{\mu,A}}`
        qtilde (Bool): When ``True`` perform the calculation using the alternative test statistic, :math:`\tilde{q}`, as defined in Equation (62) of `arXiv:1007.1727`_

    Returns:
        Tuple of Floats: The :math:`p`-values for the signal + background, background only, and signal only hypotheses respectivley
    """
    tensorlib, _ = get_backend()
    if not qtilde:  # qmu
        nullval = sqrtqmu_v
        altval = -(sqrtqmuA_v - sqrtqmu_v)
    else:  # qtilde

        def _true_case():
            nullval = sqrtqmu_v
            altval = -(sqrtqmuA_v - sqrtqmu_v)
            return nullval, altval

        def _false_case():
            qmu = tensorlib.power(sqrtqmu_v, 2)
            qmu_A = tensorlib.power(sqrtqmuA_v, 2)
            nullval = (qmu + qmu_A) / (2 * sqrtqmuA_v)
            altval = (qmu - qmu_A) / (2 * sqrtqmuA_v)
            return nullval, altval

        nullval, altval = tensorlib.conditional(
            (sqrtqmu_v < sqrtqmuA_v)[0], _true_case, _false_case
        )
    CLsb = 1 - tensorlib.normal_cdf(nullval)
    CLb = 1 - tensorlib.normal_cdf(altval)
    CLs = CLsb / CLb
    return CLsb, CLb, CLs


def pvals_from_teststat_expected(sqrtqmuA_v, nsigma=0):
    r"""
    Computes the expected :math:`p`-values CLsb, CLb and CLs for data corresponding to a given percentile of the alternate hypothesis.

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

    tensorlib, _ = get_backend()
    CLsb = tensorlib.normal_cdf(nsigma - sqrtqmuA_v)
    CLb = tensorlib.normal_cdf(nsigma)
    CLs = CLsb / CLb
    return CLsb, CLb, CLs
