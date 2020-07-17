"""PyTorch Optimizer Backend."""

from .. import get_backend
import torch


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
    tensorlib, _ = get_backend()

    if do_grad:

        def func(pars):
            pars = tensorlib.astensor(pars)
            pars.requires_grad = True
            constrained_pars = tv.stitch([fixed_values_tensor, pars])
            constr_nll = objective(constrained_pars, data, pdf)
            grad = torch.autograd.grad(constr_nll, pars)[0]
            return constr_nll.detach().numpy(), grad

    else:

        def func(pars):
            pars = tensorlib.astensor(pars)
            constrained_pars = tv.stitch([fixed_values_tensor, pars])
            constr_nll = objective(constrained_pars, data, pdf)
            return constr_nll

    return func
