from scipy.optimize import minimize
import logging

log = logging.getLogger(__name__)


class scipy_optimizer(object):
    def __init__(self, **kwargs):
        self.maxiter = kwargs.get('maxiter', 100000)

    def unconstrained_bestfit(self, objective, data, pdf, init_pars, par_bounds):
        # The Global Fit
        result = minimize(
            objective,
            init_pars,
            method='SLSQP',
            args=(data, pdf),
            bounds=par_bounds,
            options=dict(maxiter=self.maxiter),
        )
        try:
            assert result.success
        except AssertionError:
            log.error(result)
            raise
        return result.x

    def constrained_bestfit(
        self, objective, constrained_mu, data, pdf, init_pars, par_bounds
    ):
        # The Fit Conditions on a specific POI value
        cons = {'type': 'eq', 'fun': lambda v: v[pdf.config.poi_index] - constrained_mu}
        result = minimize(
            objective,
            init_pars,
            constraints=cons,
            method='SLSQP',
            args=(data, pdf),
            bounds=par_bounds,
            options=dict(maxiter=self.maxiter),
        )
        try:
            assert result.success
        except AssertionError:
            log.error(result)
            raise
        return result.x
