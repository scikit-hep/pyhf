"""PyTorch Optimizer Backend."""

from .. import get_backend, default_backend
from ..tensor.common import _TensorViewer
from .autodiff import AutoDiffOptimizerMixin
import torch


class pytorch_optimizer(AutoDiffOptimizerMixin):
    """PyTorch Optimizer Backend."""

    def setup_minimize(
        self, objective, data, pdf, init_pars, par_bounds, fixed_vals=None
    ):
        """
        Prepare Minimization for AutoDiff-Optimizer.

        Args:
            objective: objective function
            data: observed data
            pdf: model
            init_pars: initial parameters
            par_bounds: parameter boundaries
            fixed_vals: fixed parameter values

        """
        tensorlib, _ = get_backend()
        all_idx = default_backend.astensor(range(pdf.config.npars), dtype='int')
        all_init = default_backend.astensor(init_pars)

        fixed_vals = fixed_vals or []
        fixed_values = [x[1] for x in fixed_vals]
        fixed_idx = [x[0] for x in fixed_vals]

        variable_idx = [x for x in all_idx if x not in fixed_idx]
        variable_init = all_init[variable_idx]
        variable_bounds = [par_bounds[i] for i in variable_idx]

        tv = _TensorViewer([fixed_idx, variable_idx])

        data = tensorlib.astensor(data)
        fixed_values_tensor = tensorlib.astensor(fixed_values, dtype='float')

        def func(pars):
            pars = tensorlib.astensor(pars)
            pars.requires_grad = True
            constrained_pars = tv.stitch([fixed_values_tensor, pars])
            constr_nll = objective(constrained_pars, data, pdf)
            grad = torch.autograd.grad(constr_nll, pars)[0]
            return constr_nll.detach().numpy(), grad

        return tv, fixed_values_tensor, func, variable_init, variable_bounds
