from .. import get_backend
from .mle import fixed_poi_fit, fit
from ..exceptions import UnspecifiedPOI

import logging

log = logging.getLogger(__name__)


def _qmu_like(mu, data, pdf, init_pars, par_bounds, fixed_params):
    """
    Clipped version of _tmu_like where the returned test statistic
    is 0 if muhat > 0 else tmu_like_stat.

    If the lower bound of the POI is 0 this automatically implments
    qmu_tilde. Otherwise this is qmu (no tilde).
    """
    tensorlib, optimizer = get_backend()
    tmu_like_stat, (_, muhatbhat) = _tmu_like(
        mu, data, pdf, init_pars, par_bounds, fixed_params, return_fitted_pars=True
    )
    qmu_like_stat = tensorlib.where(
        muhatbhat[pdf.config.poi_index] > mu, tensorlib.astensor(0.0), tmu_like_stat
    )
    return qmu_like_stat


def _tmu_like(
    mu, data, pdf, init_pars, par_bounds, fixed_params, return_fitted_pars=False
):
    """
    Basic Profile Likelihood test statistic.

    If the lower bound of the POI is 0 this automatically implments
    tmu_tilde. Otherwise this is tmu (no tilde).
    """
    tensorlib, optimizer = get_backend()
    mubhathat, fixed_poi_fit_lhood_val = fixed_poi_fit(
        mu, data, pdf, init_pars, par_bounds, fixed_params, return_fitted_val=True
    )
    muhatbhat, unconstrained_fit_lhood_val = fit(
        data, pdf, init_pars, par_bounds, fixed_params, return_fitted_val=True
    )
    log_likelihood_ratio = fixed_poi_fit_lhood_val - unconstrained_fit_lhood_val
    tmu_like_stat = tensorlib.astensor(
        tensorlib.clip(log_likelihood_ratio, 0.0, max_value=None)
    )
    if return_fitted_pars:
        return tmu_like_stat, (mubhathat, muhatbhat)
    return tmu_like_stat


def qmu(mu, data, pdf, init_pars, par_bounds, fixed_params):
    r"""
    The test statistic, :math:`q_{\mu}`, for establishing an upper
    limit on the strength parameter, :math:`\mu`, as defiend in
    Equation (14) in :xref:`arXiv:1007.1727`

    .. math::
       :nowrap:

       \begin{equation}
          q_{\mu} = \left\{\begin{array}{ll}
          -2\ln\lambda\left(\mu\right), &\hat{\mu} < \mu,\\
          0, & \hat{\mu} > \mu
          \end{array}\right.
        \end{equation}

    where :math:`\lambda\left(\mu\right)` is the profile likelihood ratio as defined in Equation (7)

    .. math::

       \lambda\left(\mu\right) = \frac{L\left(\mu, \hat{\hat{\boldsymbol{\theta}}}\right)}{L\left(\hat{\mu}, \hat{\boldsymbol{\theta}}\right)}\,.

    Example:
        >>> import pyhf
        >>> pyhf.set_backend("numpy")
        >>> model = pyhf.simplemodels.hepdata_like(
        ...     signal_data=[12.0, 11.0], bkg_data=[50.0, 52.0], bkg_uncerts=[3.0, 7.0]
        ... )
        >>> observations = [51, 48]
        >>> data = pyhf.tensorlib.astensor(observations + model.config.auxdata)
        >>> test_mu = 1.0
        >>> init_pars = model.config.suggested_init()
        >>> par_bounds = model.config.suggested_bounds()
        >>> par_bounds[model.config.poi_index] = [-10.0, 10.0]
        >>> fixed_params = model.config.suggested_fixed()
        >>> pyhf.infer.test_statistics.qmu(test_mu, data, model, init_pars, par_bounds, fixed_params)
        array(3.9549891)

    Args:
        mu (Number or Tensor): The signal strength parameter
        data (Tensor): The data to be considered
        pdf (~pyhf.pdf.Model): The HistFactory statistical model used in the likelihood ratio calculation
        init_pars (`list`): Values to initialize the model parameters at for the fit
        par_bounds (`list` of `list`\s or `tuple`\s): The extrema of values the model parameters are allowed to reach in the fit
        fixed_params (`list`): Parameters held constant in the fit

    Returns:
        Float: The calculated test statistic, :math:`q_{\mu}`
    """
    if pdf.config.poi_index is None:
        raise UnspecifiedPOI(
            'No POI is defined. A POI is required for profile likelihood based test statistics.'
        )
    if par_bounds[pdf.config.poi_index][0] == 0:
        log.warning(
            'qmu test statistic used for fit configuration with POI bounded at zero.\n'
            + 'Use the qmu_tilde test statistic (pyhf.infer.test_statistics.qmu_tilde) instead.'
        )
    return _qmu_like(mu, data, pdf, init_pars, par_bounds, fixed_params)


def qmu_tilde(mu, data, pdf, init_pars, par_bounds, fixed_params):
    r"""
    The test statistic, :math:`\tilde{q}_{\mu}`, for establishing an upper
    limit on the strength parameter, :math:`\mu`, for models with
    bounded POI, as defiend in Equation (16) in :xref:`arXiv:1007.1727`

    .. math::
       :nowrap:

       \begin{equation}
          \tilde{q}_{\mu} = \left\{\begin{array}{ll}
          -2\ln\tilde{\lambda}\left(\mu\right), &\hat{\mu} < \mu,\\
          0, & \hat{\mu} > \mu
          \end{array}\right.
        \end{equation}

    where :math:`\tilde{\lambda}\left(\mu\right)` is the constrained profile likelihood ratio as defined in Equation (10)

    .. math::
       :nowrap:

       \begin{equation}
          \tilde{\lambda}\left(\mu\right) = \left\{\begin{array}{ll}
          \frac{L\left(\mu, \hat{\hat{\boldsymbol{\theta}}}(\mu)\right)}{L\left(\hat{\mu}, \hat{\hat{\boldsymbol{\theta}}}(0)\right)}, &\hat{\mu} < 0,\\
          \frac{L\left(\mu, \hat{\hat{\boldsymbol{\theta}}}(\mu)\right)}{L\left(\hat{\mu}, \hat{\boldsymbol{\theta}}\right)}, &\hat{\mu} \geq 0.
          \end{array}\right.
        \end{equation}

    Example:
        >>> import pyhf
        >>> pyhf.set_backend("numpy")
        >>> model = pyhf.simplemodels.hepdata_like(
        ...     signal_data=[12.0, 11.0], bkg_data=[50.0, 52.0], bkg_uncerts=[3.0, 7.0]
        ... )
        >>> observations = [51, 48]
        >>> data = pyhf.tensorlib.astensor(observations + model.config.auxdata)
        >>> test_mu = 1.0
        >>> init_pars = model.config.suggested_init()
        >>> par_bounds = model.config.suggested_bounds()
        >>> fixed_params = model.config.suggested_fixed()
        >>> pyhf.infer.test_statistics.qmu_tilde(test_mu, data, model, init_pars, par_bounds, fixed_params)
        array(3.93824492)

    Args:
        mu (Number or Tensor): The signal strength parameter
        data (Tensor): The data to be considered
        pdf (~pyhf.pdf.Model): The statistical model adhering to the schema model.json
        init_pars (`list`): Values to initialize the model parameters at for the fit
        par_bounds (`list` of `list`\s or `tuple`\s): The extrema of values the model parameters are allowed to reach in the fit
        fixed_params (`list`): Parameters held constant in the fit

    Returns:
        Float: The calculated test statistic, :math:`\tilde{q}_{\mu}`
    """
    if pdf.config.poi_index is None:
        raise UnspecifiedPOI(
            'No POI is defined. A POI is required for profile likelihood based test statistics.'
        )
    if par_bounds[pdf.config.poi_index][0] != 0:
        log.warning(
            'qmu_tilde test statistic used for fit configuration with POI not bounded at zero.\n'
            + 'Use the qmu test statistic (pyhf.infer.test_statistics.qmu) instead.'
        )
    return _qmu_like(mu, data, pdf, init_pars, par_bounds, fixed_params)


def tmu(mu, data, pdf, init_pars, par_bounds, fixed_params):
    r"""
    The test statistic, :math:`t_{\mu}`, for establishing a two-sided
    interval on the strength parameter, :math:`\mu`, as defiend in Equation (8)
    in :xref:`arXiv:1007.1727`

    .. math::

       t_{\mu} = -2\ln\lambda\left(\mu\right)

    where :math:`\lambda\left(\mu\right)` is the profile likelihood ratio as defined in Equation (7)

    .. math::

       \lambda\left(\mu\right) = \frac{L\left(\mu, \hat{\hat{\boldsymbol{\theta}}}\right)}{L\left(\hat{\mu}, \hat{\boldsymbol{\theta}}\right)}\,.

    Example:
        >>> import pyhf
        >>> pyhf.set_backend("numpy")
        >>> model = pyhf.simplemodels.hepdata_like(
        ...     signal_data=[12.0, 11.0], bkg_data=[50.0, 52.0], bkg_uncerts=[3.0, 7.0]
        ... )
        >>> observations = [51, 48]
        >>> data = pyhf.tensorlib.astensor(observations + model.config.auxdata)
        >>> test_mu = 1.0
        >>> init_pars = model.config.suggested_init()
        >>> par_bounds = model.config.suggested_bounds()
        >>> par_bounds[model.config.poi_index] = [-10.0, 10.0]
        >>> fixed_params = model.config.suggested_fixed()
        >>> pyhf.infer.test_statistics.tmu(test_mu, data, model, init_pars, par_bounds, fixed_params)
        array(3.9549891)

    Args:
        mu (Number or Tensor): The signal strength parameter
        data (Tensor): The data to be considered
        pdf (~pyhf.pdf.Model): The statistical model adhering to the schema model.json
        init_pars (`list`): Values to initialize the model parameters at for the fit
        par_bounds (`list` of `list`\s or `tuple`\s): The extrema of values the model parameters are allowed to reach in the fit
        fixed_params (`list`): Parameters held constant in the fit

    Returns:
        Float: The calculated test statistic, :math:`t_{\mu}`
    """
    if pdf.config.poi_index is None:
        raise UnspecifiedPOI(
            'No POI is defined. A POI is required for profile likelihood based test statistics.'
        )
    if par_bounds[pdf.config.poi_index][0] == 0:
        log.warning(
            'tmu test statistic used for fit configuration with POI bounded at zero.\n'
            + 'Use the tmu_tilde test statistic (pyhf.infer.test_statistics.tmu_tilde) instead.'
        )
    return _tmu_like(mu, data, pdf, init_pars, par_bounds, fixed_params)


def tmu_tilde(mu, data, pdf, init_pars, par_bounds, fixed_params):
    r"""
    The test statistic, :math:`\tilde{t}_{\mu}`, for establishing a two-sided
    interval on the strength parameter, :math:`\mu`, for models with
    bounded POI, as defiend in Equation (11) in :xref:`arXiv:1007.1727`

    .. math::

       \tilde{t}_{\mu} = -2\ln\tilde{\lambda}\left(\mu\right)

    where :math:`\tilde{\lambda}\left(\mu\right)` is the constrained profile likelihood ratio as defined in Equation (10)

    .. math::
       :nowrap:

       \begin{equation}
          \tilde{\lambda}\left(\mu\right) = \left\{\begin{array}{ll}
          \frac{L\left(\mu, \hat{\hat{\boldsymbol{\theta}}}(\mu)\right)}{L\left(\hat{\mu}, \hat{\hat{\boldsymbol{\theta}}}(0)\right)}, &\hat{\mu} < 0,\\
          \frac{L\left(\mu, \hat{\hat{\boldsymbol{\theta}}}(\mu)\right)}{L\left(\hat{\mu}, \hat{\boldsymbol{\theta}}\right)}, &\hat{\mu} \geq 0.
          \end{array}\right.
        \end{equation}

    Example:
        >>> import pyhf
        >>> pyhf.set_backend("numpy")
        >>> model = pyhf.simplemodels.hepdata_like(
        ...     signal_data=[12.0, 11.0], bkg_data=[50.0, 52.0], bkg_uncerts=[3.0, 7.0]
        ... )
        >>> observations = [51, 48]
        >>> data = pyhf.tensorlib.astensor(observations + model.config.auxdata)
        >>> test_mu = 1.0
        >>> init_pars = model.config.suggested_init()
        >>> par_bounds = model.config.suggested_bounds()
        >>> fixed_params = model.config.suggested_fixed()
        >>> pyhf.infer.test_statistics.tmu_tilde(test_mu, data, model, init_pars, par_bounds, fixed_params)
        array(3.93824492)

    Args:
        mu (Number or Tensor): The signal strength parameter
        data (Tensor): The data to be considered
        pdf (~pyhf.pdf.Model): The statistical model adhering to the schema model.json
        init_pars (`list`): Values to initialize the model parameters at for the fit
        par_bounds (`list` of `list`\s or `tuple`\s): The extrema of values the model parameters are allowed to reach in the fit
        fixed_params (`list`): Parameters held constant in the fit

    Returns:
        Float: The calculated test statistic, :math:`\tilde{t}_{\mu}`
    """
    if pdf.config.poi_index is None:
        raise UnspecifiedPOI(
            'No POI is defined. A POI is required for profile likelihood based test statistics.'
        )
    if par_bounds[pdf.config.poi_index][0] != 0:
        log.warning(
            'tmu_tilde test statistic used for fit configuration with POI not bounded at zero.\n'
            + 'Use the tmu test statistic (pyhf.infer.test_statistics.tmu) instead.'
        )
    return _tmu_like(mu, data, pdf, init_pars, par_bounds, fixed_params)
