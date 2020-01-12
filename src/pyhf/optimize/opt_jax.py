"""PyTorch Optimizer Backend."""

from .. import get_backend, default_backend
from ..tensor.common import _TensorViewer
from .autodiff import AutoDiffOptimizerMixin
import jax


class jax_optimizer(AutoDiffOptimizerMixin):
    """JAX Optimizer Backend."""

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
        def scalar_objective(pars,data):
            return objective(pars,data,pdf)[0]
        scalar_objective = jax.jit(scalar_objective)


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


        def final_objective(pars):
            pars = tensorlib.astensor(pars)
            constrained_pars = tv.stitch([fixed_values_tensor, pars])
            return scalar_objective(constrained_pars,data)
        jitted_objective_and_grad = jax.value_and_grad(jax.jit(final_objective))
        def func(pars):
            a,b = jitted_objective_and_grad(pars)
            return a.reshape((1,)),b
        return tv, fixed_values_tensor, func, variable_init, variable_bounds
