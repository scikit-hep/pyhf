"""
Module for Maximum Likelihood Estimation
"""
from .. import get_backend


def loglambdav(pars, data, pdf):
    return -2 * pdf.logpdf(pars, data)


def fit(data, pdf, init_pars=None, par_bounds=None, **kwargs):
    _, opt = get_backend()
    init_pars = init_pars or pdf.config.suggested_init()
    par_bounds = par_bounds or pdf.config.suggested_bounds()
    return opt.minimize(loglambdav, data, pdf, init_pars, par_bounds, **kwargs)


def fixed_poi_fit(poi_val, data, pdf, init_pars=None, par_bounds=None, poi_index=None, **kwargs):
    _, opt = get_backend()
    init_pars = init_pars or pdf.config.suggested_init()
    par_bounds = par_bounds or pdf.config.suggested_bounds()
    return opt.minimize(
        loglambdav,
        data,
        pdf,
        init_pars,
        par_bounds,
        [(pdf.config.poi_index, poi_val)],
        **kwargs
    )
