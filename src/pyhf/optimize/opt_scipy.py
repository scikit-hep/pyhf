"""scipy.optimize-based Optimizer using finite differences."""

from scipy.optimize import minimize
import logging

log = logging.getLogger(__name__)


class scipy_optimizer(object):
    """scipy.optimize-based Optimizer using finite differences."""

    def __init__(self, **kwargs):
        """Create scipy.optimize-based Optimizer."""
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
        if return_fitted_val:
            return result.x, result.fun
        return result.x
