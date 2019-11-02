from .. import get_backend, default_backend
from ..tensor.common import _TensorViewer
from .autodiff import AutoDiffOptimizerMixin
import tensorflow as tf


def eval_func(op, argop, dataop, d):
    def func(p):
        tensorlib, _ = get_backend()
        xx = tensorlib.tolist(p) if isinstance(p, tf.Tensor) else p
        yy = tensorlib.tolist(d) if isinstance(d, tf.Tensor) else d
        value = tensorlib.session.run(op, feed_dict={argop: xx, dataop: yy})
        return value

    return func


class tflow_optimizer(AutoDiffOptimizerMixin):
    def __init__(self, *args, **kargs):
        pass

    def setup_unconstrained(self, objective, m, d, init_pars, par_bounds):
        pars = tf.placeholder(tf.float32, (m.config.npars,))
        data = tf.placeholder(tf.float32, (m.config.nmaindata + m.config.nauxdata,))
        nll = objective(pars, data, m)
        nllgrad = tf.identity(tf.gradients(nll, pars)[0])
        func = eval_func([nll, nllgrad], pars, data, d)
        return func, init_pars, par_bounds

    def setup_constrained(self, objective, poival, m, d, init_pars, par_bounds):
        tensorlib, _ = get_backend()
        data = tf.placeholder(tf.float32, (m.config.nmaindata + m.config.nauxdata,))
        idx = default_backend.astensor(range(m.config.npars), dtype='int')
        init = default_backend.astensor(init_pars)
        nuisinit = default_backend.concatenate(
            [init[: m.config.poi_index], init[m.config.poi_index + 1 :]]
        ).tolist()
        nuisidx = default_backend.concatenate(
            [idx[: m.config.poi_index], idx[m.config.poi_index + 1 :]]
        ).tolist()
        nuisbounds = [par_bounds[i] for i in nuisidx]

        poivals = tensorlib.astensor([poival], dtype='float')
        free_pars_for_constrained = tf.placeholder(tf.float32, (m.config.npars - 1,))
        tv = _TensorViewer([[m.config.poi_index], nuisidx])
        constrained_pars = tv.stitch([poivals, free_pars_for_constrained])
        constr_nll = objective(constrained_pars, data, m)
        constr_nllgrad = tf.identity(
            tf.gradients(constr_nll, free_pars_for_constrained)[0]
        )

        func = eval_func(
            [constr_nll, constr_nllgrad], free_pars_for_constrained, data, d
        )
        return func, nuisinit, nuisbounds
