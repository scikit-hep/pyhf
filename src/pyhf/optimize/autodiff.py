"""Helper Classes for use of automatic differentiation."""
from scipy.optimize import minimize
from .. import get_backend
import logging

log = logging.getLogger(__name__)


class AutoDiffOptimizerMixin(object):
    """Mixin Class to build optimizers that use automatic differentiation."""

    def __init__(self, *args, **kwargs):
        """Create Mixin for autodiff-based optimizers."""
        self.maxiter = kwargs.get('maxiter', 100000)

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
        result = minimize(func, init, method='SLSQP', bounds=bounds, jac=True,)
        try:
            assert result.success
        except AssertionError:
            log.error(result)
            raise
        nonfixed_vals = result.x
        fitted_val = result.fun
        fitted_pars = tv.stitch(
            [fixed_values_tensor, tensorlib.astensor(nonfixed_vals)]
        )
        if return_fitted_val:
            return fitted_pars, tensorlib.astensor(fitted_val)
        return fitted_pars
