"""Numpy Backend Function Shim."""

from .. import get_backend
from .. import exceptions


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
        raise exceptions.Unsupported("Numpy does not support autodifferentiation.")

    def func(pars):
        pars = tensorlib.astensor(pars)
        constrained_pars = stitch_pars(pars)
        return objective(constrained_pars, data, pdf)

    return func
