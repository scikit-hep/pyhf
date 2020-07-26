"""Tensorflow Backend Function Shim."""
from .. import get_backend
import tensorflow as tf


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
            with tf.GradientTape() as tape:
                tape.watch(pars)
                constrained_pars = stitch_pars(pars)
                constr_nll = objective(constrained_pars, data, pdf)
            # NB: tape.gradient can return a sparse gradient (tf.IndexedSlices)
            # when tf.gather is used and this needs to be converted back to a
            # tensor to be usable as a value
            grad = tape.gradient(constr_nll, pars)
            return constr_nll.numpy()[0], tf.convert_to_tensor(grad)

    else:

        def func(pars):
            pars = tensorlib.astensor(pars)
            constrained_pars = stitch_pars(pars)
            return objective(constrained_pars, data, pdf)[0]

    return func
