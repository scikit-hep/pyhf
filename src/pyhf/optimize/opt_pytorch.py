"""PyTorch Backend Function Shim."""

from .. import get_backend
import torch


def wrap_objective(
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
            pars = tensorlib.astensor(pars)
            pars.requires_grad = True
            constrained_pars = build_pars(pars)
            constr_nll = objective(constrained_pars, data, pdf)
            grad = torch.autograd.grad(constr_nll, pars)[0]
            return constr_nll.detach().numpy(), grad

    else:

        def func(pars):
            pars = tensorlib.astensor(pars)
            constrained_pars = build_pars(pars)
            constr_nll = objective(constrained_pars, data, pdf)
            return constr_nll

    return func
