"""Module for Maximum Likelihood Estimation."""
from .. import get_backend
from ..exceptions import UnspecifiedPOI


def nll(pars, data, pdf):
    r"""
    The negative log-likelihood of the model parameters, :math:`\left(\mu, \boldsymbol{\theta}\right)`, given the observed data.
    It is used in the calculation of the test statistic, :math:`t_{\mu}`, as defiend in Equation (8) in :xref:`arXiv:1007.1727`

    .. math::

       t_{\mu} = -2\ln\lambda\left(\mu\right)

    where :math:`\lambda\left(\mu\right)` is the profile likelihood ratio as defined in Equation (7)

    .. math::

       \lambda\left(\mu\right) = \frac{L\left(\mu, \hat{\hat{\boldsymbol{\theta}}}\right)}{L\left(\hat{\mu}, \hat{\boldsymbol{\theta}}\right)}\,.

    It serves as the objective function to minimize in :func:`~pyhf.infer.mle.fit`
    and :func:`~pyhf.infer.mle.fixed_poi_fit`.

    Example:
        >>> import pyhf
        >>> pyhf.set_backend("numpy")
        >>> model = pyhf.simplemodels.hepdata_like(
        ...     signal_data=[12.0, 11.0], bkg_data=[50.0, 52.0], bkg_uncerts=[3.0, 7.0]
        ... )
        >>> observations = [51, 48]
        >>> data = pyhf.tensorlib.astensor(observations + model.config.auxdata)
        >>> parameters = model.config.suggested_init()  # nominal parameters
        >>> nll = pyhf.infer.mle.nll(parameters, data, model)
        >>> nll
        array([15.38762717])
        >>> -model.logpdf(parameters, data) == nll
        array([ True])

    Args:
        pars (:obj:`tensor`): The parameters of the HistFactory model
        data (:obj:`tensor`): The data to be considered
        pdf (~pyhf.pdf.Model): The statistical model adhering to the schema model.json

    Returns:
        Tensor: The negative log-likelihood, :math:`-\ln L\left(\mu, \boldsymbol{\theta}\right)`
    """
    return -pdf.logpdf(pars, data)


def _validate_fit_inputs(init_pars, par_bounds, fixed_params):
    for par_idx, (value, bound) in enumerate(zip(init_pars, par_bounds)):
        if not (bound[0] <= value <= bound[1]):
            raise ValueError(
                f"fit initialization parameter (index: {par_idx}, value: {value}) lies outside of its bounds: {bound}"
                + "\nTo correct this adjust the initialization parameter values in the model spec or those given"
                + "\nas arguments to pyhf.infer.fit. If this value is intended, adjust the range of the parameter"
                + "\nbounds."
            )


def fit(data, pdf, init_pars=None, par_bounds=None, fixed_params=None, **kwargs):
    r"""
    Run a maximum likelihood fit.
    This is done by minimizing the objective function :func:`~pyhf.infer.mle.nll`
    of the model parameters given the observed data.
    This is used to produce the maximal likelihood :math:`L\left(\hat{\mu}, \hat{\boldsymbol{\theta}}\right)`
    in the profile likelihood ratio in Equation (7) in :xref:`arXiv:1007.1727`

    .. math::

       \lambda\left(\mu\right) = \frac{L\left(\mu, \hat{\hat{\boldsymbol{\theta}}}\right)}{L\left(\hat{\mu}, \hat{\boldsymbol{\theta}}\right)}


    .. note::

        :func:`nll` is the objective function given to the optimizer and
        is returned evaluated at the best fit model parameters when the optional
        kwarg ``return_fitted_val`` is ``True``.

    Example:
        >>> import pyhf
        >>> pyhf.set_backend("numpy")
        >>> model = pyhf.simplemodels.hepdata_like(
        ...     signal_data=[12.0, 11.0], bkg_data=[50.0, 52.0], bkg_uncerts=[3.0, 7.0]
        ... )
        >>> observations = [51, 48]
        >>> data = pyhf.tensorlib.astensor(observations + model.config.auxdata)
        >>> bestfit_pars, nll = pyhf.infer.mle.fit(data, model, return_fitted_val=True)
        >>> bestfit_pars
        array([0.        , 1.00305155, 0.96267465])
        >>> nll
        array(12.4919676)
        >>> -model.logpdf(bestfit_pars, data) == nll
        array([ True])

    Args:
        data (:obj:`tensor`): The data
        pdf (~pyhf.pdf.Model): The statistical model adhering to the schema model.json
        init_pars (:obj:`list`): Values to initialize the model parameters at for the fit
        par_bounds (:obj:`list` of :obj:`list`\s or :obj:`tuple`\s): The extrema of values the model parameters are allowed to reach in the fit
        fixed_params (:obj:`list`): Parameters to be held constant in the fit.
        kwargs: Keyword arguments passed through to the optimizer API

    Returns:
        See optimizer API

    """
    _, opt = get_backend()
    init_pars = init_pars or pdf.config.suggested_init()
    par_bounds = par_bounds or pdf.config.suggested_bounds()
    fixed_params = fixed_params or pdf.config.suggested_fixed()

    _validate_fit_inputs(init_pars, par_bounds, fixed_params)

    # get fixed vals from the model
    fixed_vals = [
        (index, init)
        for index, (init, is_fixed) in enumerate(zip(init_pars, fixed_params))
        if is_fixed
    ]

    return opt.minimize(nll, data, pdf, init_pars, par_bounds, fixed_vals, **kwargs)


def fixed_poi_fit(
    poi_val, data, pdf, init_pars=None, par_bounds=None, fixed_params=None, **kwargs
):
    r"""
    Run a maximum likelihood fit with the POI value fixed.
    This is done by minimizing the objective function of :func:`~pyhf.infer.mle.nll`
    of the model parameters given the observed data, for a given fixed value of :math:`\mu`.
    This is used to produce the constrained maximal likelihood for the given :math:`\mu`,
    :math:`L\left(\mu, \hat{\hat{\boldsymbol{\theta}}}\right)`, in the profile
    likelihood ratio in Equation (7) in :xref:`arXiv:1007.1727`

    .. math::

       \lambda\left(\mu\right) = \frac{L\left(\mu, \hat{\hat{\boldsymbol{\theta}}}\right)}{L\left(\hat{\mu}, \hat{\boldsymbol{\theta}}\right)}

    .. note::

        :func:`nll` is the objective function given to the optimizer and
        is returned evaluated at the best fit model parameters when the optional
        kwarg ``return_fitted_val`` is ``True``.

    Example:
        >>> import pyhf
        >>> pyhf.set_backend("numpy")
        >>> model = pyhf.simplemodels.hepdata_like(
        ...     signal_data=[12.0, 11.0], bkg_data=[50.0, 52.0], bkg_uncerts=[3.0, 7.0]
        ... )
        >>> observations = [51, 48]
        >>> data = pyhf.tensorlib.astensor(observations + model.config.auxdata)
        >>> test_poi = 1.0
        >>> bestfit_pars, nll = pyhf.infer.mle.fixed_poi_fit(
        ...     test_poi, data, model, return_fitted_val=True
        ... )
        >>> bestfit_pars
        array([1.        , 0.97226646, 0.87552889])
        >>> nll
        array(14.46109013)
        >>> -model.logpdf(bestfit_pars, data) == nll
        array([ True])

    Args:
        data: The data
        pdf (~pyhf.pdf.Model): The statistical model adhering to the schema model.json
        init_pars (:obj:`list`): Values to initialize the model parameters at for the fit
        par_bounds (:obj:`list` of :obj:`list`\s or :obj:`tuple`\s): The extrema of values the model parameters are allowed to reach in the fit
        fixed_params (:obj:`list`): Parameters to be held constant in the fit.
        kwargs: Keyword arguments passed through to the optimizer API

    Returns:
        See optimizer API

    """
    if pdf.config.poi_index is None:
        raise UnspecifiedPOI(
            'No POI is defined. A POI is required to fit with a fixed POI.'
        )

    init_pars = [*(init_pars or pdf.config.suggested_init())]
    fixed_params = [*(fixed_params or pdf.config.suggested_fixed())]

    init_pars[pdf.config.poi_index] = poi_val
    fixed_params[pdf.config.poi_index] = True

    return fit(data, pdf, init_pars, par_bounds, fixed_params, **kwargs)
