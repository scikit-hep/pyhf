"""Module for Maximum Likelihood Estimation."""
from .. import get_backend
from ..exceptions import UnspecifiedPOI


def twice_nll(pars, data, pdf):
    """
    Twice the negative Log-Likelihood.

    Args:
        data (`tensor`): The data
        pdf (~pyhf.pdf.Model): The statistical model adhering to the schema model.json

    Returns:
        Twice the negative log likelihood.

    """
    return -2 * pdf.logpdf(pars, data)


def fit(data, pdf, init_pars=None, par_bounds=None, **kwargs):
    """
    Run a unconstrained maximum likelihood fit.

    .. note::

        :func:`twice_nll` is the objective function given to the optimizer and
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
        >>> bestfit_pars, twice_nll = pyhf.infer.mle.fit(data, model, return_fitted_val=True)
        >>> bestfit_pars
        array([0.        , 1.0030512 , 0.96266961])
        >>> twice_nll
        array(24.98393521)
        >>> -2 * model.logpdf(bestfit_pars, data) == twice_nll
        array([ True])

    Args:
        data (`tensor`): The data
        pdf (~pyhf.pdf.Model): The statistical model adhering to the schema model.json
        init_pars (`list`): Values to initialize the model parameters at for the fit
        par_bounds (`list` of `list`\s or `tuple`\s): The extrema of values the model parameters are allowed to reach in the fit
        kwargs: Keyword arguments passed through to the optimizer API

    Returns:
        See optimizer API

    """
    _, opt = get_backend()
    init_pars = init_pars or pdf.config.suggested_init()
    par_bounds = par_bounds or pdf.config.suggested_bounds()
    return opt.minimize(twice_nll, data, pdf, init_pars, par_bounds, **kwargs)


def fixed_poi_fit(poi_val, data, pdf, init_pars=None, par_bounds=None, **kwargs):
    """
    Run a maximum likelihood fit with the POI value fixed.

    .. note::

        :func:`twice_nll` is the objective function given to the optimizer and
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
        >>> bestfit_pars, twice_nll = pyhf.infer.mle.fixed_poi_fit(
        ...     test_poi, data, model, return_fitted_val=True
        ... )
        >>> bestfit_pars
        array([1.        , 0.97224597, 0.87553894])
        >>> twice_nll
        array(28.92218013)
        >>> -2 * model.logpdf(bestfit_pars, data) == twice_nll
        array([ True])

    Args:
        data: The data
        pdf (~pyhf.pdf.Model): The statistical model adhering to the schema model.json
        init_pars (`list`): Values to initialize the model parameters at for the fit
        par_bounds (`list` of `list`\s or `tuple`\s): The extrema of values the model parameters are allowed to reach in the fit
        kwargs: Keyword arguments passed through to the optimizer API

    Returns:
        See optimizer API

    """
    if pdf.config.poi_index is None:
        raise UnspecifiedPOI(
            'No POI is defined. A POI is required to fit with a fixed POI.'
        )
    _, opt = get_backend()
    init_pars = init_pars or pdf.config.suggested_init()
    par_bounds = par_bounds or pdf.config.suggested_bounds()
    return opt.minimize(
        twice_nll,
        data,
        pdf,
        init_pars,
        par_bounds,
        [(pdf.config.poi_index, poi_val)],
        **kwargs,
    )
