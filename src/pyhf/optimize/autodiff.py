import scipy
from .. import default_backend
from .. import get_backend


class AutoDiffOptimizerMixin(object):
    def unconstrained_bestfit(self, objective, data, pdf, init_pars, par_bounds):
        func, init, bounds = self.setup_unconstrained(
            objective, data, pdf, init_pars, par_bounds
        )
        fitresult = scipy.optimize.minimize(
            func, init, method='SLSQP', jac=True, bounds=bounds
        )
        unconstr_pars = fitresult.x
        tensorlib, _ = get_backend()
        return tensorlib.astensor(unconstr_pars)

    def constrained_bestfit(
        self, objective, constrained_mu, data, pdf, init_pars, par_bounds
    ):
        poival = constrained_mu
        func, init, bounds = self.setup_constrained(
            objective, poival, data, pdf, init_pars, par_bounds
        )
        fitresult = scipy.optimize.minimize(
            func, init, method='SLSQP', jac=True, bounds=bounds
        )
        constr_pars = default_backend.concatenate(
            [
                fitresult.x[: pdf.config.poi_index],
                [poival],
                fitresult.x[pdf.config.poi_index :],
            ]
        )
        tensorlib, _ = get_backend()
        return tensorlib.astensor(constr_pars)
