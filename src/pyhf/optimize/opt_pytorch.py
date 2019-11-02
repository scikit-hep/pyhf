from .. import get_backend, default_backend
from ..tensor.common import _TensorViewer
from .autodiff import AutoDiffOptimizerMixin
import torch


class pytorch_optimizer(AutoDiffOptimizerMixin):
    def __init__(self, *args, **kargs):
        pass

    def setup_unconstrained(self, objective, m, d, init_pars, par_bounds):
        def func(p):
            tensorlib, _ = get_backend()
            pars = tensorlib.astensor(p)
            pars.requires_grad = True
            r = objective(pars, d, m)
            grad = torch.autograd.grad(r, pars)[0]
            return r.detach().numpy(), grad

        return func, init_pars, par_bounds

    def setup_constrained(self, objective, poival, m, d, init_pars, par_bounds):
        tensorlib, _ = get_backend()
        idx = default_backend.astensor(range(m.config.npars), dtype='int')
        init = default_backend.astensor(init_pars)
        nuisinit = default_backend.concatenate(
            [init[: m.config.poi_index], init[m.config.poi_index + 1 :]]
        ).tolist()
        nuisidx = default_backend.concatenate(
            [idx[: m.config.poi_index], idx[m.config.poi_index + 1 :]]
        ).tolist()
        nuisbounds = [par_bounds[i] for i in nuisidx]
        tv = _TensorViewer([[m.config.poi_index], nuisidx])

        data = tensorlib.astensor(d)
        poivals = tensorlib.astensor([poival], dtype='float')

        def func(p):
            pars = tensorlib.astensor(p)
            pars.requires_grad = True
            constrained_pars = tv.stitch([poivals, pars])
            constr_nll = objective(constrained_pars, data, m)
            grad = torch.autograd.grad(constr_nll, pars)[0]
            return constr_nll.detach().numpy(), grad

        return func, nuisinit, nuisbounds
