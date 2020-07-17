"""Tensorflow Optimizer Backend."""
from .. import get_backend, default_backend
from .. import exceptions
from ..tensor.common import _TensorViewer


def _get_tensor_shim():
    tensorlib, _ = get_backend()
    if tensorlib.name == 'numpy':
        from .opt_numpy import make_func as numpy_shim

        return numpy_shim
    elif tensorlib.name == 'tensorflow':
        try:
            from .opt_tflow import make_func as tflow_shim

            return tflow_shim
        except ImportError as e:
            raise exceptions.ImportBackendError(
                "There was a problem importing TensorFlow. The pytorch backend cannot be used.",
                e,
            )
    elif tensorlib.name == 'pytorch':
        try:
            from .opt_pytorch import make_func as pytorch_shim

            return pytorch_shim
        except ImportError as e:
            raise exceptions.ImportBackendError(
                "There was a problem importing PyTorch. The pytorch backend cannot be used.",
                e,
            )
    elif tensorlib.name == 'jax':
        try:
            from .opt_jax import make_func as jax_shim

            return jax_shim
        except ImportError as e:
            raise exceptions.ImportBackendError(
                "There was a problem importing JAX. The pytorch backend cannot be used.",
                e,
            )
    else:
        raise ValueError(f'No optimizer shim for {tensorlib.name}.')


def shim(objective, data, pdf, init_pars, par_bounds, fixed_vals=None, do_grad=False):
    """
    Prepare Minimization for Optimizer.

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

    # NB: need to pass in fixed_idx, variable_idx for jax to jit correctly
    func = _get_tensor_shim()(
        objective,
        data,
        pdf,
        tv,
        fixed_values_tensor,
        fixed_idx=fixed_idx,
        variable_idx=variable_idx,
        do_grad=do_grad,
    )

    return tv, fixed_values_tensor, func, variable_init, variable_bounds
