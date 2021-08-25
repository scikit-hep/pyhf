"""Numpy Backend Function Shim."""

from pyhf import get_backend
from pyhf import exceptions


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

    if do_grad:
        raise exceptions.Unsupported("Numpy does not support autodifferentiation.")

    def func(pars):
        pars = tensorlib.astensor(pars)
        constrained_pars = stitch_pars(pars)
        return objective(constrained_pars, data, pdf)[0]

    return func
