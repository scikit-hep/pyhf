from scipy.optimize import minimize
import logging

log = logging.getLogger(__name__)


class scipy_optimizer(object):
    def __init__(self, **kwargs):
        self.maxiter = kwargs.get('maxiter', 100000)

    def minimize(self, objective, data, pdf, init_pars, par_bounds, fixed_vals=None):
        fixed_vals = fixed_vals or []
        indices = [i for i, _ in fixed_vals]
        values = [v for _, v in fixed_vals]
        constraints = [{'type': 'eq', 'fun': lambda v: v[indices] - values}]
        result = minimize(
            objective,
            init_pars,
            constraints=constraints,
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

    def unconstrained_bestfit(self, objective, data, pdf, init_pars, par_bounds):
        return self.minimize(objective, data, pdf, init_pars, par_bounds)

    def constrained_bestfit(
        self, objective, constrained_mu, data, pdf, init_pars, par_bounds
    ):
        return self.minimize(
            objective,
            data,
            pdf,
            init_pars,
            par_bounds,
            [(pdf.config.poi_index, constrained_mu)],
        )
