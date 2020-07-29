"""PyTorch Backend Function Shim."""

from .. import get_backend
import torch


def wrap_objective(objective, data, pdf, stitch_pars, do_grad=False, jit_pieces=None):
    """
    Wrap the objective function for the minimization.

    Args:
        objective (`func`): objective function
        data (`list`): observed data
        pdf (~pyhf.pdf.Model): The statistical model adhering to the schema model.json
        stitch_pars (`func`): callable that stitches parameters, see :func:`pyhf.optimize.common.shim`.
        do_grad (`bool`): enable autodifferentiation mode. Default is off.

    Returns:
        objective_and_grad (`func`): tensor backend wrapped objective,gradient pair
    """

    tensorlib, _ = get_backend()

    if do_grad:

        def func(pars):
            pars = tensorlib.astensor(pars)
            pars.requires_grad = True
            constrained_pars = stitch_pars(pars)
            constr_nll = objective(constrained_pars, data, pdf)
            grad = torch.autograd.grad(constr_nll, pars)[0]
            return constr_nll.detach().numpy()[0], grad

    else:

        def func(pars):
            pars = tensorlib.astensor(pars)
            constrained_pars = stitch_pars(pars)
            constr_nll = objective(constrained_pars, data, pdf)
            return constr_nll[0]

    return func
