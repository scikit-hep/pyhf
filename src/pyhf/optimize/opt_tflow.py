"""Tensorflow Backend Function Shim."""
from .. import get_backend
import tensorflow as tf


def make_func(
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
            with tf.GradientTape() as tape:
                tape.watch(pars)
                constrained_pars = build_pars(pars)
                constr_nll = objective(constrained_pars, data, pdf)
            grad = tape.gradient(constr_nll, pars).values
            return constr_nll.numpy(), grad

    else:

        def func(pars):
            pars = tensorlib.astensor(pars)
            constrained_pars = build_pars(pars)
            return objective(constrained_pars, data, pdf)

    return func
