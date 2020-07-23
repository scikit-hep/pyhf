"""JAX Backend Function Shim."""

from .. import get_backend
from ..tensor.common import _TensorViewer
import jax
import logging

log = logging.getLogger(__name__)


def _final_objective(pars, data, fixed_values, fixed_idx, variable_idx, objective, pdf):
    log.debug('jitting function')
    tensorlib, _ = get_backend()
    tv = _TensorViewer([fixed_idx, variable_idx])
    pars = tensorlib.astensor(pars)
    constrained_pars = tv.stitch(
        [tensorlib.astensor(fixed_values, dtype='float'), pars]
    )
    return objective(constrained_pars, data, pdf)[0]


_jitted_objective_and_grad = jax.jit(
    jax.value_and_grad(_final_objective, argnums=0), static_argnums=(3, 4, 5, 6)
)

_jitted_objective = jax.jit(_final_objective, static_argnums=(3, 4, 5, 6))


def wrap_objective(objective, data, pdf, stitch_pars, do_grad=False, jit_pieces=None):
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
    # NB: tuple arguments that need to be hashable (static_argnums)
    if do_grad:

        def func(pars):
            # need to conver to tuple to make args hashable
            return _jitted_objective_and_grad(
                pars,
                data,
                jit_pieces['fixed_values'],
                tuple(jit_pieces['fixed_idx']),
                tuple(jit_pieces['variable_idx']),
                objective,
                pdf,
            )

    else:

        def func(pars):
            # need to conver to tuple to make args hashable
            return _jitted_objective(
                pars,
                data,
                jit_pieces['fixed_values'],
                tuple(jit_pieces['fixed_idx']),
                tuple(jit_pieces['variable_idx']),
                objective,
                pdf,
            )

    return func
