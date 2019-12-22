"""Helper Classes for use of automatic differentiation."""
import scipy
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
    ):
        """
        Find Function Parameters that minimize the Objective.

        Returns:
            bestfit parameters
        
        """
        tensorlib, _ = get_backend()
        tv, fixed_values_tensor, func, init, bounds = self.setup_minimize(
            objective, data, pdf, init_pars, par_bounds, fixed_vals
        )
        fitresult = scipy.optimize.minimize(
            func, init, method='SLSQP', jac=True, bounds=bounds
        )
        nonfixed_vals = fitresult.x
        fitted_val = fitresult.fun
        fitted_pars = tv.stitch(
            [fixed_values_tensor, tensorlib.astensor(nonfixed_vals)]
        )
        if return_fitted_val:
            return fitted_pars, tensorlib.astensor(fitted_val)
        return fitted_pars
