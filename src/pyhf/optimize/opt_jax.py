"""JAX Backend Function Shim."""

from .. import get_backend
from ..tensor.common import _TensorViewer
import jax
import logging

log = logging.getLogger(__name__)


def to_inf(x, bounds):
    tensorlib, _ = get_backend()
    lo, hi = bounds.T
    return tensorlib.arcsin(2 * (x - lo) / (hi - lo) - 1)


def to_bnd(x, bounds):
    tensorlib, _ = get_backend()
    lo, hi = bounds.T
    return lo + 0.5 * (hi - lo) * (tensorlib.sin(x) + 1)


def _final_objective(
    pars,
    data,
    fixed_values,
    fixed_idx,
    variable_idx,
    do_stitch,
    objective,
    pdf,
    par_bounds,
):
    log.debug('jitting function')
    tensorlib, _ = get_backend()
    pars = tensorlib.astensor(pars)

    pars = to_bnd(pars, par_bounds)

    if do_stitch:
        tv = _TensorViewer([fixed_idx, variable_idx])
        constrained_pars = tv.stitch(
            [tensorlib.astensor(fixed_values, dtype='float'), pars]
        )
    else:
        constrained_pars = pars
    return objective(constrained_pars, data, pdf)[0]


_jitted_objective_and_grad = jax.jit(
    jax.value_and_grad(_final_objective, argnums=0), static_argnums=(3, 4, 5, 6, 7)
)

_jitted_objective = jax.jit(_final_objective, static_argnums=(3, 4, 5, 6, 7))


def wrap_objective(objective, data, pdf, stitch_pars, do_grad=False, jit_pieces=None):
    """
    Wrap the objective function for the minimization.

    Args:
        objective (:obj:`func`): objective function
        data (:obj:`list`): observed data
        pdf (~pyhf.pdf.Model): The statistical model adhering to the schema model.json
        stitch_pars (:obj:`func`): callable that stitches parameters, see :func:`pyhf.optimize.common.shim`.
        do_grad (:obj:`bool`): enable autodifferentiation mode. Default is off.

    Returns:
        objective_and_grad (:obj:`func`): tensor backend wrapped objective,gradient pair
    """
    tensorlib, _ = get_backend()
    # NB: tuple arguments that need to be hashable (static_argnums)
    if do_grad:

        def func(pars):
            # need to conver to tuple to make args hashable
            result = _jitted_objective_and_grad(
                pars,
                data,
                jit_pieces['fixed_values'],
                tuple(jit_pieces['fixed_idx']),
                tuple(jit_pieces['variable_idx']),
                jit_pieces['do_stitch'],
                objective,
                pdf,
                jit_pieces['par_bounds'],
            )
            return result

    else:

        def func(pars):
            # need to conver to tuple to make args hashable
            return _jitted_objective(
                pars,
                data,
                jit_pieces['fixed_values'],
                tuple(jit_pieces['fixed_idx']),
                tuple(jit_pieces['variable_idx']),
                jit_pieces['do_stitch'],
                objective,
                pdf,
                jit_pieces['par_bounds'],
            )

    return func
