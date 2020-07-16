"""Helper Classes for use of automatic differentiation."""
import iminuit
import numpy as np
from .. import get_backend


class AutoDiffOptimizerMixin(object):
    """Mixin Class to build optimizers that use automatic differentiation."""

    def __init__(*args, **kwargs):
        """Create Mixin for autodiff-based optimizers."""

    def minimize(
        self,
        objective,
        data,
        pdf,
        init_pars,
        par_bounds,
        fixed_vals=None,
        return_fitted_val=False,
        return_uncertainties=False,
    ):
        """
        Find Function Parameters that minimize the Objective.

        Returns:
            bestfit parameters

        """
        tensorlib, _ = get_backend()
        tv, fixed_values_tensor, func_and_grad, init, bounds = self.setup_minimize(
            objective, data, pdf, init_pars, par_bounds, fixed_vals
        )
        func = lambda pars: func_and_grad(pars)[0]
        grad = lambda pars: func_and_grad(pars)[1]

        fitresult = iminuit.minimize(
            func, init, method='SLSQP', jac=grad, bounds=bounds
        )
        assert fitresult

        mm = fitresult.minuit
        if return_uncertainties:
            bestfit_pars = np.asarray([(v, mm.errors[k]) for k, v in mm.values.items()])
        else:
            bestfit_pars = np.asarray([v for k, v in mm.values.items()])
        bestfit_value = mm.fval
        if return_fitted_val:
            return bestfit_pars, bestfit_value
        return bestfit_pars
