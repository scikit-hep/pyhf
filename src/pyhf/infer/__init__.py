"""Inference for Statistical Models."""

from . import utils
from .. import get_backend


def hypotest(
    poi_test,
    data,
    pdf,
    init_pars=None,
    par_bounds=None,
    fixed_params=None,
    calctype="asymptotics",
    return_tail_probs=False,
    return_expected=False,
    return_expected_set=False,
    **kwargs,
):
    r"""
    Compute :math:`p`-values and test statistics for a single value of the parameter of interest.

    See :py:class:`~pyhf.infer.calculators.AsymptoticCalculator` and :py:class:`~pyhf.infer.calculators.ToyCalculator` on additional keyword arguments to be specified.

    Example:
        >>> import pyhf
        >>> pyhf.set_backend("numpy")
        >>> model = pyhf.simplemodels.hepdata_like(
        ...     signal_data=[12.0, 11.0], bkg_data=[50.0, 52.0], bkg_uncerts=[3.0, 7.0]
        ... )
        >>> observations = [51, 48]
        >>> data = pyhf.tensorlib.astensor(observations + model.config.auxdata)
        >>> mu_test = 1.0
        >>> CLs_obs, CLs_exp_band = pyhf.infer.hypotest(
        ...     mu_test, data, model, return_expected_set=True, test_stat="qtilde"
        ... )
        >>> CLs_obs
        array(0.0525151)
        >>> CLs_exp_band
        [array(0.0026063), array(0.01382019), array(0.06445367), array(0.2352575), array(0.5730375)]

    Args:
        poi_test (Number or Tensor): The value of the parameter of interest (POI)
        data (Number or Tensor): The data considered
        pdf (~pyhf.pdf.Model): The statistical model adhering to the schema ``model.json``
        init_pars (:obj:`tensor`): The initial parameter values to be used for minimization
        par_bounds (:obj:`tensor`): The parameter value bounds to be used for minimization
        fixed_params (:obj:`tensor`): Whether to fix the parameter to the init_pars value during minimization
        calctype (:obj:`str`): The calculator to create. Choose either 'asymptotics' (default) or 'toybased'.
        return_tail_probs (:obj:`bool`): Bool for returning :math:`\mathrm{CL}_{s+b}` and :math:`\mathrm{CL}_{b}`
        return_expected (:obj:`bool`): Bool for returning :math:`\mathrm{CL}_{\mathrm{exp}}`
        return_expected_set (:obj:`bool`): Bool for returning the :math:`(-2,-1,0,1,2)\sigma` :math:`\mathrm{CL}_{\mathrm{exp}}` --- the "Brazil band"

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

    calc = utils.create_calculator(
        calctype,
        data,
        pdf,
        init_pars,
        par_bounds,
        fixed_params,
        **kwargs,
    )

    teststat = calc.teststatistic(poi_test)
    sig_plus_bkg_distribution, bkg_only_distribution = calc.distributions(poi_test)

    tb, _ = get_backend()
    CLsb_obs, CLb_obs, CLs_obs = tuple(
        tb.astensor(pvalue)
        for pvalue in calc.pvalues(
            teststat, sig_plus_bkg_distribution, bkg_only_distribution
        )
    )
    CLsb_exp, CLb_exp, CLs_exp = calc.expected_pvalues(
        sig_plus_bkg_distribution, bkg_only_distribution
    )

    is_q0 = kwargs.get('test_stat', 'qtilde') == 'q0'

    _returns = [CLsb_obs if is_q0 else CLs_obs]
    if return_tail_probs:
        if is_q0:
            _returns.append([CLb_obs])
        else:
            _returns.append([CLsb_obs, CLb_obs])

    pvalues_exp_band = [
        tb.astensor(pvalue) for pvalue in (CLsb_exp if is_q0 else CLs_exp)
    ]
    if return_expected_set:
        if return_expected:
            _returns.append(tb.astensor(pvalues_exp_band[2]))
        _returns.append(pvalues_exp_band)
    elif return_expected:
        _returns.append(tb.astensor(pvalues_exp_band[2]))
    # Enforce a consistent return type of the observed CLs
    return tuple(_returns) if len(_returns) > 1 else _returns[0]


from . import intervals  # noqa: F401

__all__ = ["hypotest"]
