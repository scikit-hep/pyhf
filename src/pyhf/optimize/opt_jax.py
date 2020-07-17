"""JAX Backend Function Shim."""

from .. import get_backend
from ..tensor.common import _TensorViewer
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

_jitted_objective = jax.jit(_final_objective, static_argnums=(3, 4, 5, 6))


def make_func(
    objective,
    data,
    pdf,
    tv,
    fixed_values_tensor,
    fixed_idx=[],
    variable_idx=[],
    do_grad=False,
):
    """
    Wrap the objective function for the minimization.

    Args:
        objective: objective function
        data: observed data
        pdf: model
        init_pars: initial parameters
        par_bounds: parameter boundaries
        fixed_vals: fixed parameter values

    Returns:
        func: tensor backend wrapped function,gradient pair
    """
    tensorlib, _ = get_backend()

    if do_grad:

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

    else:

        def func(pars):
            # need to conver to tuple to make args hashable
            return _jitted_objective(
                pars,
                data,
                fixed_values_tensor,
                pdf,
                objective,
                tuple(fixed_idx),
                tuple(variable_idx),
            )

    return func
