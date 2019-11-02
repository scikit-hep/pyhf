import scipy
from .. import default_backend


class AutoDiffOptimizerMixin(object):
    def unconstrained_bestfit(self, objective, data, pdf, init_pars, par_bounds):
        for_unconstr = self.setup_unconstrained(
            objective, pdf, data, init_pars, par_bounds
        )
        fitresult = scipy.optimize.minimize(
            for_unconstr[0],
            for_unconstr[1],
            method='SLSQP',
            jac=True,
            bounds=for_unconstr[2],
        )
        unconstr_pars = fitresult.x
        return unconstr_pars

    def constrained_bestfit(
        self, objective, constrained_mu, data, pdf, init_pars, par_bounds
    ):
        poival = constrained_mu
        for_constr = self.setup_constrained(
            objective, poival, pdf, data, init_pars, par_bounds
        )
        fitresult = scipy.optimize.minimize(
            for_constr[0], for_constr[1], method='SLSQP', jac=True, bounds=for_constr[2]
        )
        constr_pars = default_backend.concatenate(
            [
                fitresult.x[: pdf.config.poi_index],
                [poival],
                fitresult.x[pdf.config.poi_index :],
            ]
        )
        return constr_pars
