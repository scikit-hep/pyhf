"""JAX Backend Function Shim."""

from .. import get_backend
import jax


def _final_objective(objective, data, pdf, stitch_pars, pars):
    tensorlib, _ = get_backend()
    pars = tensorlib.astensor(pars)
    constrained_pars = stitch_pars(pars)
    return objective(constrained_pars, data, pdf)[0]


_jitted_objective_and_grad = jax.jit(
    jax.value_and_grad(_final_objective, argnums=4), static_argnums=(0, 2, 3)
)

_jitted_objective = jax.jit(_final_objective, static_argnums=(0, 2, 3))


def wrap_objective(objective, data, pdf, stitch_pars, do_grad=False):
    """
    Wrap the objective function for the minimization.

    Args:
        objective (`func`): objective function
        data (`list`): observed data
        pdf (`pyhf.pdf.Model`): model
        stitch_pars (`func`): callable that stitches parameters, see :func:`pyhf.optimize.common.shim`.
        do_grad (`bool`): enable autodifferentiation mode. Default is off.

    Returns:
        objective_and_grad (`func`): tensor backend wrapped objective,gradient pair
    """
    tensorlib, _ = get_backend()

    if do_grad:

        def func(pars):
            # need to conver to tuple to make args hashable
            return _jitted_objective_and_grad(objective, data, pdf, stitch_pars, pars,)

    else:

        def func(pars):
            # need to conver to tuple to make args hashable
            return _jitted_objective(objective, data, pdf, stitch_pars, pars,)

    return func
