"""
Module for Maximum Likelihood Estimation
"""
from .. import get_backend

def loglambdav(pars, data, pdf):
    return -2 * pdf.logpdf(pars, data)

def floating_poi_mle(data, pdf, init_pars, par_bounds, **kwargs):
    _,opt = get_backend()
    return opt.minimize(
            loglambdav,
            data,
            pdf,
            init_pars,
            par_bounds,
            **kwargs
    )

def fixed_poi_mle(constrained_mu, data, pdf, init_pars, par_bounds, **kwargs):
    _,opt = get_backend()
    return opt.minimize(
            loglambdav,
            data,
            pdf,
            init_pars,
            par_bounds,
            [(pdf.config.poi_index, constrained_mu)],
            **kwargs
    )