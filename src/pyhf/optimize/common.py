"""Common Backend Shim to prepare minimization for optimizer."""
from .. import get_backend, default_backend
from ..tensor.common import _TensorViewer


def _get_tensor_shim():
    """
    A shim-retriever to lazy-retrieve the necessary shims as needed.

    Because pyhf.tensor is a lazy-retriever for the backends, we can be sure
    that tensorlib is imported correctly.
    """
    tensorlib, _ = get_backend()
    if tensorlib.name == 'numpy':
        from .opt_numpy import make_func as numpy_shim

        return numpy_shim

    if tensorlib.name == 'tensorflow':
        from .opt_tflow import make_func as tflow_shim

        return tflow_shim

    if tensorlib.name == 'pytorch':
        from .opt_pytorch import make_func as pytorch_shim

        return pytorch_shim

    if tensorlib.name == 'jax':
        from .opt_jax import make_func as jax_shim

        return jax_shim
    raise ValueError(f'No optimizer shim for {tensorlib.name}.')


def shim(
    objective,
    data,
    pdf,
    init_pars,
    par_bounds,
    fixed_vals=None,
    do_grad=False,
    do_stitch=False,
):
    """
    Prepare Minimization for Optimizer.

    Args:
        objective: objective function
        data: observed data
        pdf: model
        init_pars: initial parameters
        par_bounds: parameter boundaries
        fixed_vals: fixed parameter values

    Returns:
        tv: tensor viewer
        fixed_values_tensor: constant parameters in the fit
        func: tensor backend wrapped function,gradient pair
        variable_init: initializations for minimizer
        variable_bounds: bounds for minimizer
    """
    tensorlib, _ = get_backend()

    if do_stitch:
        all_init = default_backend.astensor(init_pars)
        all_idx = default_backend.astensor(range(pdf.config.npars), dtype='int')

        fixed_vals = fixed_vals or []
        fixed_values = [x[1] for x in fixed_vals]
        fixed_idx = [x[0] for x in fixed_vals]

        variable_idx = [x for x in all_idx if x not in fixed_idx]
        variable_init = default_backend.tolist(all_init[variable_idx])
        variable_bounds = [par_bounds[i] for i in variable_idx]

        tv = _TensorViewer([fixed_idx, variable_idx])
        # NB: this is a closure, tensorlib needs to be accessed at a different point in time
        def stitch_pars(pars, stitch_with=fixed_values):
            tb, _ = get_backend()
            return tv.stitch([tb.astensor(fixed_values, dtype='float'), pars])

    else:
        tv = None
        variable_init = init_pars
        variable_bounds = par_bounds
        stitch_pars = lambda pars, stitch_with=None: pars

    objective_and_grad = _get_tensor_shim()(
        objective, tensorlib.astensor(data), pdf, stitch_pars, do_grad=do_grad,
    )

    if do_grad:
        wrapped_objective = lambda pars: objective_and_grad(pars)[0]
        jac = lambda pars: objective_and_grad(pars)[1]
    else:
        wrapped_objective = objective_and_grad
        jac = None

    return stitch_pars, wrapped_objective, jac, variable_init, variable_bounds
