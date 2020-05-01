"""Module for Maximum Likelihood Estimation."""
from .. import get_backend


def twice_nll(pars, data, pdf):
    """
    Twice the negative Log-Likelihood.

    Args:
        data (`tensor`): the data
        pdf (~pyhf.pdf.Model): The statistical model adhering to the schema model.json

    Returns:
        Twice the negative log likelihood.

    """
    return -2 * pdf.logpdf(pars, data)


def fit(data, pdf, init_pars=None, par_bounds=None, **kwargs):
    """
    Run a unconstrained maximum likelihood fit.

    Args:
        data (`tensor`): The data
        pdf (~pyhf.pdf.Model): The statistical model adhering to the schema model.json
        init_pars (`list`): Values to initialize the model paramters at for the fit
        par_bounds (`list` of `list`\s or `tuple`\s): The extrema of values the model paramters are allowed to reach in the fit
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
    Run a maximum likelihood fit with the POI value fixzed.

    Args:
        data: The data
        pdf (~pyhf.pdf.Model): The statistical model adhering to the schema model.json
        init_pars (`list`): Values to initialize the model paramters at for the fit
        par_bounds (`list` of `list`\s or `tuple`\s): The extrema of values the model paramters are allowed to reach in the fit
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
