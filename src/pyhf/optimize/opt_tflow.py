from .. import get_backend, default_backend
from ..tensor.common import _TensorViewer
from .autodiff import AutoDiffOptimizerMixin
import tensorflow as tf


def eval_func(op, argop, dataop, data):
    def func(pars):
        tensorlib, _ = get_backend()
        pars_as_list = tensorlib.tolist(pars) if isinstance(pars, tf.Tensor) else pars
        data_as_list = tensorlib.tolist(data) if isinstance(data, tf.Tensor) else data
        value = tensorlib.session.run(
            op, feed_dict={argop: pars_as_list, dataop: data_as_list}
        )
        return value

    return func


class tflow_optimizer(AutoDiffOptimizerMixin):
    def __init__(self, *args, **kargs):
        pass

    def setup_unconstrained(self, objective, data, pdf, init_pars, par_bounds):
        pars_placeholder = tf.placeholder(tf.float32, (pdf.config.npars,))
        data_placeholder = tf.placeholder(
            tf.float32, (pdf.config.nmaindata + pdf.config.nauxdata,)
        )
        nll = objective(pars_placeholder, data_placeholder, pdf)
        nllgrad = tf.identity(tf.gradients(nll, pars_placeholder)[0])
        func = eval_func([nll, nllgrad], pars_placeholder, data_placeholder, data)
        return func, init_pars, par_bounds

    def setup_constrained(self, objective, poival, data, pdf, init_pars, par_bounds):
        tensorlib, _ = get_backend()
        data_placeholder = tf.placeholder(
            tf.float32, (pdf.config.nmaindata + pdf.config.nauxdata,)
        )
        idx = default_backend.astensor(range(pdf.config.npars), dtype='int')
        init = default_backend.astensor(init_pars)
        nuisinit = default_backend.concatenate(
            [init[: pdf.config.poi_index], init[pdf.config.poi_index + 1 :]]
        ).tolist()
        nuisidx = default_backend.concatenate(
            [idx[: pdf.config.poi_index], idx[pdf.config.poi_index + 1 :]]
        ).tolist()
        nuisbounds = [par_bounds[i] for i in nuisidx]

        poivals = tensorlib.astensor([poival], dtype='float')
        free_pars_for_constrained = tf.placeholder(tf.float32, (pdf.config.npars - 1,))
        tv = _TensorViewer([[pdf.config.poi_index], nuisidx])
        constrained_pars = tv.stitch([poivals, free_pars_for_constrained])
        constr_nll = objective(constrained_pars, data_placeholder, pdf)
        constr_nllgrad = tf.identity(
            tf.gradients(constr_nll, free_pars_for_constrained)[0]
        )

        func = eval_func(
            [constr_nll, constr_nllgrad],
            free_pars_for_constrained,
            data_placeholder,
            data,
        )
        return func, nuisinit, nuisbounds
