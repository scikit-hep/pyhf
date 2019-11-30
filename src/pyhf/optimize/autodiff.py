import scipy
from .. import get_backend


class AutoDiffOptimizerMixin(object):
    def minimize(self, objective, data, pdf, init_pars, par_bounds, fixed_vals=None):
        tensorlib, _ = get_backend()
        tv, fixed_values_tensor, func, init, bounds = self.setup_minimize(
            objective, data, pdf, init_pars, par_bounds, fixed_vals
        )
        fitresult = scipy.optimize.minimize(
            func, init, method='SLSQP', jac=True, bounds=bounds
        )
        nonfixed_vals = fitresult.x
        return tv.stitch([fixed_values_tensor, tensorlib.astensor(nonfixed_vals)])

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
