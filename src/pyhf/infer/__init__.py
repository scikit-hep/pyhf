"""Inference for Statistical Models."""

from .test_statistics import qmu
from .. import get_backend
from .calculators import AsymptoticCalculator


def hypotest(
    poi_test,
    data,
    pdf,
    init_pars=None,
    par_bounds=None,
    fixed_params=None,
    qtilde=False,
    **kwargs,
):
    r"""
    Compute :math:`p`-values and test statistics for a single value of the parameter of interest.

    Example:
        >>> import pyhf
        >>> pyhf.set_backend("numpy")
        >>> model = pyhf.simplemodels.hepdata_like(
        ...     signal_data=[12.0, 11.0], bkg_data=[50.0, 52.0], bkg_uncerts=[3.0, 7.0]
        ... )
        >>> observations = [51, 48]
        >>> data = pyhf.tensorlib.astensor(observations + model.config.auxdata)
        >>> test_poi = 1.0
        >>> CLs_obs, CLs_exp_band = pyhf.infer.hypotest(
        ...     test_poi, data, model, qtilde=True, return_expected_set=True
        ... )
        >>> CLs_obs
        array(0.05251497)
        >>> CLs_exp_band
        [array(0.00260626), array(0.01382005), array(0.06445321), array(0.23525644), array(0.57303621)]

    Args:
        poi_test (Number or Tensor): The value of the parameter of interest (POI)
        data (Number or Tensor): The data considered
        pdf (~pyhf.pdf.Model): The statistical model adhering to the schema ``model.json``
        init_pars (`tensor`): The initial parameter values to be used for minimization
        par_bounds (`tensor`): The parameter value bounds to be used for minimization
        fixed_params (`tensor`): Whether to fix the parameter to the init_pars value during minimization
        qtilde (Bool): When ``True`` perform the calculation using the alternative
         test statistic, :math:`\tilde{q}_{\mu}`, as defined under the Wald
         approximation in Equation (62) of :xref:`arXiv:1007.1727`.

    Keyword Args:
        return_tail_probs (bool): Bool for returning :math:`\mathrm{CL}_{s+b}` and :math:`\mathrm{CL}_{b}`
        return_expected (bool): Bool for returning :math:`\mathrm{CL}_{\mathrm{exp}}`
        return_expected_set (bool): Bool for returning the :math:`(-2,-1,0,1,2)\sigma` :math:`\mathrm{CL}_{\mathrm{exp}}` --- the "Brazil band"

    Returns:
        Tuple of Floats and lists of Floats:

            - :math:`\mathrm{CL}_{s}`: The modified :math:`p`-value compared to
              the given threshold :math:`\alpha`, typically taken to be :math:`0.05`,
              defined in :xref:`arXiv:1007.1727` as

            .. math::

                \mathrm{CL}_{s} = \frac{\mathrm{CL}_{s+b}}{\mathrm{CL}_{b}} = \frac{p_{s+b}}{1-p_{b}}

            to protect against excluding signal models in which there is little
            sensitivity. In the case that :math:`\mathrm{CL}_{s} \leq \alpha`
            the given signal model is excluded.

            - :math:`\left[\mathrm{CL}_{s+b}, \mathrm{CL}_{b}\right]`: The
              signal + background model hypothesis :math:`p`-value

            .. math::

                \mathrm{CL}_{s+b} = p_{s+b}
                = p\left(q \geq q_{\mathrm{obs}}\middle|s+b\right)
                = \int\limits_{q_{\mathrm{obs}}}^{\infty} f\left(q\,\middle|s+b\right)\,dq
                = 1 - F\left(q_{\mathrm{obs}}(\mu)\,\middle|\mu'\right)

            and 1 minus the background only model hypothesis :math:`p`-value

            .. math::

                \mathrm{CL}_{b} = 1- p_{b}
                = p\left(q \geq q_{\mathrm{obs}}\middle|b\right)
                = 1 - \int\limits_{-\infty}^{q_{\mathrm{obs}}} f\left(q\,\middle|b\right)\,dq
                = 1 - F\left(q_{\mathrm{obs}}(\mu)\,\middle|0\right)

            for signal strength :math:`\mu` and model hypothesis signal strength
            :math:`\mu'`, where the cumulative density functions
            :math:`F\left(q(\mu)\,\middle|\mu'\right)` are given by Equations (57)
            and (65) of :xref:`arXiv:1007.1727` for upper-limit-like test
            statistic :math:`q \in \{q_{\mu}, \tilde{q}_{\mu}\}`.
            Only returned when ``return_tail_probs`` is ``True``.

            .. note::

                The definitions of the :math:`\mathrm{CL}_{s+b}` and
                :math:`\mathrm{CL}_{b}` used are based on profile likelihood
                ratio test statistics.
                This procedure is common in the LHC-era, but differs from
                procedures used in the LEP and Tevatron eras, as briefly
                discussed in :math:`\S` 3.8 of :xref:`arXiv:1007.1727`.

            - :math:`\mathrm{CL}_{s,\mathrm{exp}}`: The expected :math:`\mathrm{CL}_{s}`
              value corresponding to the test statistic under the background
              only hypothesis :math:`\left(\mu=0\right)`.
              Only returned when ``return_expected`` is ``True``.

            - :math:`\mathrm{CL}_{s,\mathrm{exp}}` band: The set of expected
              :math:`\mathrm{CL}_{s}` values corresponding to the median
              significance of variations of the signal strength from the
              background only hypothesis :math:`\left(\mu=0\right)` at
              :math:`(-2,-1,0,1,2)\sigma`.
              That is, the :math:`p`-values that satisfy Equation (89) of
              :xref:`arXiv:1007.1727`

            .. math::

                \mathrm{band}_{N\sigma} = \mu' + \sigma\,\Phi^{-1}\left(1-\alpha\right) \pm N\sigma

            for :math:`\mu'=0` and :math:`N \in \left\{-2, -1, 0, 1, 2\right\}`.
            These values define the boundaries of an uncertainty band sometimes
            referred to as the "Brazil band".
            Only returned when ``return_expected_set`` is ``True``.

    """
    init_pars = init_pars or pdf.config.suggested_init()
    par_bounds = par_bounds or pdf.config.suggested_bounds()
    fixed_params = fixed_params or pdf.config.suggested_fixed()

    calc = AsymptoticCalculator(
        data, pdf, init_pars, par_bounds, fixed_params, qtilde=qtilde
    )
    teststat = calc.teststatistic(poi_test)
    sig_plus_bkg_distribution, b_only_distribution = calc.distributions(poi_test)

    CLsb = sig_plus_bkg_distribution.pvalue(teststat)
    CLb = b_only_distribution.pvalue(teststat)
    CLs = CLsb / CLb

    tensorlib, _ = get_backend()
    # Ensure that all CL values are 0-d tensors
    CLsb, CLb, CLs = (
        tensorlib.astensor(CLsb),
        tensorlib.astensor(CLb),
        tensorlib.astensor(CLs),
    )

    _returns = [CLs]
    if kwargs.get('return_tail_probs'):
        _returns.append([CLsb, CLb])
    if kwargs.get('return_expected_set'):
        CLs_exp = []
        for n_sigma in [2, 1, 0, -1, -2]:

            expected_bonly_teststat = b_only_distribution.expected_value(n_sigma)

            CLs = sig_plus_bkg_distribution.pvalue(
                expected_bonly_teststat
            ) / b_only_distribution.pvalue(expected_bonly_teststat)
            CLs_exp.append(tensorlib.astensor(CLs))
        if kwargs.get('return_expected'):
            _returns.append(CLs_exp[2])
        _returns.append(CLs_exp)
    elif kwargs.get('return_expected'):
        n_sigma = 0
        expected_bonly_teststat = b_only_distribution.expected_value(n_sigma)

        CLs = sig_plus_bkg_distribution.pvalue(
            expected_bonly_teststat
        ) / b_only_distribution.pvalue(expected_bonly_teststat)
        _returns.append(tensorlib.astensor(CLs))
    # Enforce a consistent return type of the observed CLs
    return tuple(_returns) if len(_returns) > 1 else _returns[0]


__all__ = ['qmu', 'hypotest']
