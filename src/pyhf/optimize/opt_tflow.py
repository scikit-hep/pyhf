"""Tensorflow Optimizer Backend."""
from .. import get_backend, default_backend
from ..tensor.common import _TensorViewer
from .autodiff import AutoDiffOptimizerMixin
import tensorflow as tf


def _eval_func(op, argop, dataop, data):
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
    """Tensorflow Optimizer Backend."""

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

        data_placeholder = tf.placeholder(
            tensorlib.dtypemap['float'], (pdf.config.nmaindata + pdf.config.nauxdata,)
        )
        variable_pars_placeholder = tf.placeholder(
            tensorlib.dtypemap['float'], (pdf.config.npars - len(fixed_vals),)
        )

        tv = _TensorViewer([fixed_idx, variable_idx])

        fixed_values_tensor = tensorlib.astensor(fixed_values, dtype='float')

        full_pars = tv.stitch([fixed_values_tensor, variable_pars_placeholder])

        nll = objective(full_pars, data_placeholder, pdf)
        nllgrad = tf.identity(tf.gradients(nll, variable_pars_placeholder)[0])

        func = _eval_func(
            [nll, nllgrad], variable_pars_placeholder, data_placeholder, data,
        )

        return tv, fixed_values_tensor, func, variable_init, variable_bounds
