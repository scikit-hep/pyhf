"""scipy.optimize-based Optimizer using finite differences."""
from .. import get_backend, default_backend
from ..tensor.common import _TensorViewer

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
        tensorlib, _ = get_backend()

        all_idx = default_backend.astensor(range(pdf.config.npars), dtype='int')
        all_init = default_backend.astensor(init_pars)

        fixed_vals = fixed_vals or []
        fixed_values = [x[1] for x in fixed_vals]
        fixed_idx = [x[0] for x in fixed_vals]

        variable_idx = [x for x in all_idx if x not in fixed_idx]
        variable_init = all_init[variable_idx]
        variable_bounds = [par_bounds[i] for i in variable_idx]

        tv = _TensorViewer([fixed_idx, variable_idx])

        data = tensorlib.astensor(data)
        fixed_values_tensor = tensorlib.astensor(fixed_values, dtype='float')

        def func(pars):
            pars = tensorlib.astensor(pars)
            constrained_pars = tv.stitch([fixed_values_tensor, pars])
            return objective(constrained_pars, data, pdf)

        result = minimize(
            func,
            variable_init,
            method='SLSQP',
            jac=False,
            bounds=variable_bounds,
            options=dict(maxiter=self.maxiter),
        )
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
            return fitted_pars, fitted_val
        return fitted_pars
