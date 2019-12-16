import scipy
from .. import get_backend


class AutoDiffOptimizerMixin(object):
    def minimize(
        self,
        objective,
        data,
        pdf,
        init_pars,
        par_bounds,
        fixed_vals=None,
        return_fval=False,
    ):
        tensorlib, _ = get_backend()
        tv, fixed_values_tensor, func, init, bounds = self.setup_minimize(
            objective, data, pdf, init_pars, par_bounds, fixed_vals
        )
        fitresult = scipy.optimize.minimize(
            func, init, method='SLSQP', jac=True, bounds=bounds
        )
        nonfixed_vals = fitresult.x
        fitted_fval = fitresult.fun
        fitted_pars = tv.stitch(
            [fixed_values_tensor, tensorlib.astensor(nonfixed_vals)]
        )
        if return_fval:
            return fitted_pars, tensorlib.astensor(fitted_fval)
        return fitted_pars
