"""JAX Backend Function Shim."""

from .. import get_backend
import jax


def _final_objective(objective, data, pdf, build_pars, pars):
    tensorlib, _ = get_backend()
    pars = tensorlib.astensor(pars)
    constrained_pars = build_pars(pars)
    return objective(constrained_pars, data, pdf)[0]


_jitted_objective_and_grad = jax.jit(
    jax.value_and_grad(_final_objective), static_argnums=(0, 2, 3)
)

_jitted_objective = jax.jit(_final_objective, static_argnums=(0, 2, 3))


def make_func(
    objective, data, pdf, build_pars, do_grad=False,
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
            return _jitted_objective_and_grad(objective, data, pdf, build_pars, pars,)

    else:

        def func(pars):
            # need to conver to tuple to make args hashable
            return _jitted_objective(objective, data, pdf, build_pars, pars,)

    return func
