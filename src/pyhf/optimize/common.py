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
        fixed_values_tensor = tensorlib.astensor(fixed_values, dtype='float')
        build_pars = lambda pars: tv.stitch([fixed_values_tensor, pars])
    else:
        tv = None
        fixed_values_tensor = tensorlib.astensor([], dtype='float')
        variable_init = init_pars
        variable_bounds = par_bounds
        build_pars = lambda pars: pars

    wrapped_objective = _get_tensor_shim()(
        objective, tensorlib.astensor(data), pdf, build_pars, do_grad=do_grad,
    )

    return tv, fixed_values_tensor, wrapped_objective, variable_init, variable_bounds
