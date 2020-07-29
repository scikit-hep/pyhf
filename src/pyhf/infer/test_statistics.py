from .. import get_backend
from .mle import fixed_poi_fit, fit
from ..exceptions import UnspecifiedPOI

import logging

log = logging.getLogger(__name__)


def _qmu_like(mu, data, pdf, init_pars, par_bounds):
    """
    Clipped version of _tmu where _qmu = 0 if muhat > 0 else _tmu

    If the lower bound of the POI is 0 this automatically implments
    qmu_tilde. Otherwise this is qmu (no tilde).
    """
    tensorlib, optimizer = get_backend()
    tmu_stat, (mubhathat, muhatbhat) = _tmu_like(
        mu, data, pdf, init_pars, par_bounds, return_fitted_pars=True
    )
    qmu = tensorlib.where(
        muhatbhat[pdf.config.poi_index] > mu, tensorlib.astensor(0.0), tmu_stat
    )
    return qmu


def _tmu_like(mu, data, pdf, init_pars, par_bounds, return_fitted_pars=False):
    """
    Basic Profile Likelihood statistic.

    If the lower bound of the POI is 0 this automatically implments
    tmu_tilde. Otherwise this is tmu (no tilde).
    """
    tensorlib, optimizer = get_backend()
    mubhathat, fixed_poi_fit_lhood_val = fixed_poi_fit(
        mu, data, pdf, init_pars, par_bounds, return_fitted_val=True
    )
    muhatbhat, unconstrained_fit_lhood_val = fit(
        data, pdf, init_pars, par_bounds, return_fitted_val=True
    )
    tmu = fixed_poi_fit_lhood_val - unconstrained_fit_lhood_val
    tmu = tensorlib.clip(tmu, 0, max_value=None)
    if return_fitted_pars:
        return tmu, (mubhathat, muhatbhat)
    return tmu


def qmu(mu, data, pdf, init_pars, par_bounds):
    r"""
    The test statistic, :math:`q_{\mu}`, for establishing an upper
    limit on the strength parameter, :math:`\mu`, as defiend in
    Equation (14) in :xref:`arXiv:1007.1727`.

    .. math::
       :nowrap:

       \begin{equation}
          q_{\mu} = \left\{\begin{array}{ll}
          -2\ln\lambda\left(\mu\right), &\hat{\mu} < \mu,\\
          0, & \hat{\mu} > \mu
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
        >>> pyhf.infer.test_statistics.qmu(test_mu, data, model, init_pars, par_bounds)
        array(3.93824492)

    Args:
        mu (Number or Tensor): The signal strength parameter
        data (Tensor): The data to be considered
        pdf (~pyhf.pdf.Model): The HistFactory statistical model used in the likelihood ratio calculation
        init_pars (`list`): Values to initialize the model parameters at for the fit
        par_bounds (`list` of `list`\s or `tuple`\s): The extrema of values the model parameters are allowed to reach in the fit

    Returns:
        Float: The calculated test statistic, :math:`q_{\mu}`
    """
    if pdf.config.poi_index is None:
        raise UnspecifiedPOI(
            'No POI is defined. A POI is required for profile likelihood based test statistics.'
        )
    if par_bounds[pdf.config.poi_index][0] == 0:
        log.warning(
            'qmu test statistic used for fit configuration with POI bounded at zero. Use qmutilde.'
        )
    return _qmu_like(mu, data, pdf, init_pars, par_bounds)


def qmu_tilde(mu, data, pdf, init_pars, par_bounds):
    r"""
    The test statistic, :math:`\tilde{q}_{\mu}`, for establishing an upper
    limit on the strength parameter, :math:`\mu` for models with
    bounded POI.

    Args:
        mu (Number or Tensor): The signal strength parameter
        data (Tensor): The data to be considered
        pdf (~pyhf.pdf.Model): The HistFactory statistical model used in the likelihood ratio calculation
        init_pars (`list`): Values to initialize the model parameters at for the fit
        par_bounds (`list` of `list`\s or `tuple`\s): The extrema of values the model parameters are allowed to reach in the fit

    Returns:
        Float: The calculated test statistic, :math:`q_{\mu}`
    """
    if pdf.config.poi_index is None:
        raise UnspecifiedPOI(
            'No POI is defined. A POI is required for profile likelihood based test statistics.'
        )
    if par_bounds[pdf.config.poi_index][0] != 0:
        log.warning(
            'qmu tilde test statistic used for fit configuration with POI not bounded at zero. Use qmu.'
        )
    return _qmu_like(mu, data, pdf, init_pars, par_bounds)


def tmu(mu, data, pdf, init_pars, par_bounds):
    r"""
    The test statistic, :math:`t_{\mu}`, for establishing an two-sided
    intervals on the strength parameter, :math:`\mu`.

    Args:
        mu (Number or Tensor): The signal strength parameter
        data (Tensor): The data to be considered
        pdf (~pyhf.pdf.Model): The HistFactory statistical model used in the likelihood ratio calculation
        init_pars (`list`): Values to initialize the model parameters at for the fit
        par_bounds (`list` of `list`\s or `tuple`\s): The extrema of values the model parameters are allowed to reach in the fit

    Returns:
        Float: The calculated test statistic, :math:`q_{\mu}`
    """
    if pdf.config.poi_index is None:
        raise UnspecifiedPOI(
            'No POI is defined. A POI is required for profile likelihood based test statistics.'
        )
    if par_bounds[pdf.config.poi_index][0] == 0:
        log.warning(
            'tmu test statistic used for fit configuration with POI bounded at zero. Use qmutilde.'
        )
    return _tmu_like(mu, data, pdf, init_pars, par_bounds)


def tmu_tilde(mu, data, pdf, init_pars, par_bounds):
    r"""
    The test statistic, :math:`t_{\mu}`, for establishing an two-sided
    intervals on the strength parameter, :math:`\mu` for models with
    bounded POI.

    Args:
        mu (Number or Tensor): The signal strength parameter
        data (Tensor): The data to be considered
        pdf (~pyhf.pdf.Model): The HistFactory statistical model used in the likelihood ratio calculation
        init_pars (`list`): Values to initialize the model parameters at for the fit
        par_bounds (`list` of `list`\s or `tuple`\s): The extrema of values the model parameters are allowed to reach in the fit

    Returns:
        Float: The calculated test statistic, :math:`q_{\mu}`
    """
    if pdf.config.poi_index is None:
        raise UnspecifiedPOI(
            'No POI is defined. A POI is required for profile likelihood based test statistics.'
        )
    if par_bounds[pdf.config.poi_index][0] != 0:
        log.warning(
            'tmu tilde test statistic used for fit configuration with POI not bounded at zero. Use tmu.'
        )
    return _tmu_like(mu, data, pdf, init_pars, par_bounds)
