"""JAX Optimizer Backend."""

from .. import get_backend, default_backend
from ..tensor.common import _TensorViewer
from .minimizer import ScipyOptimizer as Optimizer

# from .minimizer import MinuitOptimizer as Optimizer
import jax


def _final_objective(pars, data, fixed_vals, model, objective, fixed_idx, variable_idx):
    tensorlib, _ = get_backend()
    tv = _TensorViewer([fixed_idx, variable_idx])
    pars = tensorlib.astensor(pars)
    constrained_pars = tv.stitch([fixed_vals, pars])
    return objective(constrained_pars, data, model)[0]


_jitted_objective_and_grad = jax.jit(
    jax.value_and_grad(_final_objective), static_argnums=(3, 4, 5, 6)
)


class jax_optimizer(Optimizer):
    """JAX Optimizer Backend."""

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
        tensorlib, _ = get_backend()
        tv, fixed_values_tensor, func_and_grad, init, bounds = self.setup_minimize(
            objective, data, pdf, init_pars, par_bounds, fixed_vals
        )
        func = lambda pars: func_and_grad(pars)[0]
        grad = lambda pars: func_and_grad(pars)[1]

        fitresult = super(jax_optimizer, self).minimize(
            func, init, jac=grad, bounds=bounds
        )
        nonfixed_vals = fitresult.x
        fitted_val = fitresult.fun
        fitted_pars = tv.stitch(
            [fixed_values_tensor, tensorlib.astensor(nonfixed_vals)]
        )
        if return_fitted_val:
            return fitted_pars, tensorlib.astensor(fitted_val)
        return fitted_pars

    def setup_minimize(
        self, objective, data, pdf, init_pars, par_bounds, fixed_vals=None
    ):
        """
        Prepare Minimization for AutoDiff-Optimizer.

        Args:
            objective: objective function
            data: observed data
            pdf: model
            init_pars: initial parameters
            par_bounds: parameter boundaries
            fixed_vals: fixed parameter values

        """
        self._setup_minimizer(objective, data, pdf, init_pars, par_bounds, fixed_vals)
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
            # need to conver to tuple to make args hashable
            return _jitted_objective_and_grad(
                pars,
                data,
                fixed_values_tensor,
                pdf,
                objective,
                tuple(fixed_idx),
                tuple(variable_idx),
            )

        return tv, fixed_values_tensor, func, variable_init, variable_bounds
