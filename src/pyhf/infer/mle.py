"""Module for Maximum Likelihood Estimation."""
from .. import get_backend


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

    Example:
        >>> import pyhf
        >>> pyhf.set_backend("numpy")
        >>> model = pyhf.simplemodels.hepdata_like(
        ...     signal_data=[12.0, 11.0], bkg_data=[50.0, 52.0], bkg_uncerts=[3.0, 7.0]
        ... )
        >>> observations = [51, 48]
        >>> data = pyhf.tensorlib.astensor(observations + model.config.auxdata)
        >>> pyhf.infer.mle.fit(data, model, return_fitted_val=True)
        (array([0.        , 1.0030512 , 0.96266961]), 24.98393521454011)
        >>> # Run the same fit with a different optimizer
        ...
        >>> pyhf.set_backend("numpy", pyhf.optimize.minuit_optimizer(verbose=True))
        >>> best_fit_result = pyhf.infer.mle.fit(
        ...     data, model, return_fitted_val=True, return_uncertainties=True
        ... )
        ------------------------------------------------------------------
        | FCN = 24.98                   |      Ncalls=84 (84 total)      |
        | EDM = 8.09E-07 (Goal: 0.0002) |            up = 1.0            |
        ------------------------------------------------------------------
        |  Valid Min.   | Valid Param.  | Above EDM | Reached call limit |
        ------------------------------------------------------------------
        |     True      |     True      |   False   |       False        |
        ------------------------------------------------------------------
        | Hesse failed  |   Has cov.    | Accurate  | Pos. def. | Forced |
        ------------------------------------------------------------------
        |     False     |     True      |   True    |   True    | False  |
        ------------------------------------------------------------------
        >>> best_fit_pars = best_fit_result[0][:, 0]
        >>> best_fit_pars_uncert = best_fit_result[0][:, 1]
        >>> best_fit_lhood_value = best_fit_result[1]
        >>> print(best_fit_pars) # doctest: +SKIP
        [2.23857553e-07 1.00308914e+00 9.62725456e-01]
        >>> print(best_fit_pars_uncert) # doctest: +SKIP
        [1.86505494 0.05531769 0.09476047]
        >>> print(best_fit_lhood_value) # doctest: +SKIP
        24.983936012961976

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

    Example:
        >>> import pyhf
        >>> pyhf.set_backend("numpy")
        >>> model = pyhf.simplemodels.hepdata_like(
        ...     signal_data=[12.0, 11.0], bkg_data=[50.0, 52.0], bkg_uncerts=[3.0, 7.0]
        ... )
        >>> observations = [51, 48]
        >>> data = pyhf.tensorlib.astensor(observations + model.config.auxdata)
        >>> test_poi = 1.0
        >>> pyhf.infer.mle.fixed_poi_fit(test_poi, data, model, return_fitted_val=True)
        (array([1.        , 0.97224597, 0.87553894]), 28.92218013492061)

    Args:
        data: The data
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
    return opt.minimize(
        twice_nll,
        data,
        pdf,
        init_pars,
        par_bounds,
        [(pdf.config.poi_index, poi_val)],
        **kwargs,
    )
