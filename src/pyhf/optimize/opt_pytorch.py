from .. import get_backend, default_backend
from ..tensor.common import _TensorViewer
from .autodiff import AutoDiffOptimizerMixin
import torch


class pytorch_optimizer(AutoDiffOptimizerMixin):
    def __init__(self, *args, **kargs):
        pass

    def setup_unconstrained(self, objective, data, pdf, init_pars, par_bounds):
        def func(pars):
            tensorlib, _ = get_backend()
            pars = tensorlib.astensor(pars)
            pars.requires_grad = True
            r = objective(pars, data, pdf)
            grad = torch.autograd.grad(r, pars)[0]
            return r.detach().numpy(), grad

        return func, init_pars, par_bounds

    def setup_constrained(self, objective, poival, data, pdf, init_pars, par_bounds):
        tensorlib, _ = get_backend()
        idx = default_backend.astensor(range(pdf.config.npars), dtype='int')
        init = default_backend.astensor(init_pars)
        nuisinit = default_backend.concatenate(
            [init[: pdf.config.poi_index], init[pdf.config.poi_index + 1 :]]
        ).tolist()
        nuisidx = default_backend.concatenate(
            [idx[: pdf.config.poi_index], idx[pdf.config.poi_index + 1 :]]
        ).tolist()
        nuisbounds = [par_bounds[i] for i in nuisidx]
        tv = _TensorViewer([[pdf.config.poi_index], nuisidx])

        data = tensorlib.astensor(data)
        poivals = tensorlib.astensor([poival], dtype='float')

        def func(pars):
            pars = tensorlib.astensor(pars)
            pars.requires_grad = True
            constrained_pars = tv.stitch([poivals, pars])
            constr_nll = objective(constrained_pars, data, pdf)
            grad = torch.autograd.grad(constr_nll, pars)[0]
            return constr_nll.detach().numpy(), grad

        return func, nuisinit, nuisbounds
