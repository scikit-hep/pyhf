"""Common Backend Shim to prepare minimization for optimizer."""
from .. import get_backend
from ..tensor.common import _TensorViewer


def _make_post_processor(tv=None, fixed_values=None):
    """
    Construct a callable to stitch fixed paramter values into the unfixed parameters. See :func:`shim`.

    This is extracted out to be unit-tested for proper behavior.

    If ``tv`` or ``fixed_values`` are not provided, this returns the identity callable.

    Args:
        tv (~pyhf.tensor.common._TensorViewer): tensor viewer instance
        fixed_values (:obj:`list`): default set of values to stitch parameters with

    Returns:
        callable (:obj:`func`): a callable that takes nuisance parameter values as input
    """
    if tv is None or fixed_values is None:
        return lambda pars, stitch_with=None: pars

    def post_processor(pars, stitch_with=fixed_values):
        tb, _ = get_backend()
        return tv.stitch([tb.astensor(stitch_with, dtype='float'), pars])

    return post_processor


def _get_internal_objective(*args,**kwargs):
    """
    A shim-retriever to lazy-retrieve the necessary shims as needed.

    Because pyhf.tensor is a lazy-retriever for the backends, we can be sure
    that tensorlib is imported correctly.
    """
    tensorlib, _ = get_backend()
    if tensorlib.name == 'numpy':
        from .opt_numpy import wrap_objective as numpy_shim

        return numpy_shim(*args,**kwargs)

    if tensorlib.name == 'tensorflow':
        from .opt_tflow import wrap_objective as tflow_shim

        return tflow_shim(*args,**kwargs)

    if tensorlib.name == 'pytorch':
        from .opt_pytorch import wrap_objective as pytorch_shim

        return pytorch_shim(*args,**kwargs)

    if tensorlib.name == 'jax':
        from .opt_jax import wrap_objective as jax_shim

        return jax_shim(*args,**kwargs)
    raise ValueError(f'No optimizer shim for {tensorlib.name}.')


def to_inf(x,bounds):
    tensorlib, _ = get_backend()
    lo,hi = bounds.T
    return tensorlib.arcsin(2*(x-lo)/(hi-lo)-1)

def to_bnd(x,bounds):
    tensorlib, _ = get_backend()
    lo,hi = bounds.T
    return lo + 0.5*(hi-lo)*(tensorlib.sin(x) +1)


def _configure_internal_minimize(init_pars,variable_idx,do_stitch,par_bounds,fixed_idx,fixed_values):
    tensorlib, _ = get_backend()
    if do_stitch:
        all_init = tensorlib.astensor(init_pars)
        internal_init = tensorlib.tolist(
            tensorlib.gather(all_init, tensorlib.astensor(variable_idx, dtype='int'))
        )
        internal_bounds = [par_bounds[i] for i in variable_idx]
        # stitched out the fixed values, so we don't pass any to the underlying minimizer
        external_fixed_vals = []

        tv = _TensorViewer([fixed_idx, variable_idx])
        # NB: this is a closure, tensorlib needs to be accessed at a different point in time
        post_processor = _make_post_processor(tv, fixed_values)

    else:
        internal_init = init_pars
        internal_bounds = par_bounds
        external_fixed_vals = fixed_vals
        post_processor = _make_post_processor()

    internal_init = to_inf(tensorlib.astensor(internal_init),tensorlib.astensor(internal_bounds))
    def mypostprocessor(x):
        x = to_bnd(x,tensorlib.astensor(internal_bounds))
        return post_processor(x)

    no_internal_bounds = None


    kwargs =  dict(
        x0 =  internal_init,
        variable_bounds = internal_bounds,
        bounds=no_internal_bounds,
        fixed_vals=external_fixed_vals,
    )        
    return kwargs, mypostprocessor

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
        objective (:obj:`func`): objective function
        data (:obj:`list`): observed data
        pdf (~pyhf.pdf.Model): The statistical model adhering to the schema model.json
        init_pars (:obj:`list`): initial parameters
        par_bounds (:obj:`list`): parameter boundaries
        fixed_vals (:obj:`list`): fixed parameter values

    .. note::

        ``minimizer_kwargs`` is a dictionary containing

          - ``func`` (:obj:`func`): backend-wrapped ``objective`` function (potentially with gradient)
          - ``x0`` (:obj:`list`):  modified initializations for minimizer
          - ``do_grad`` (:obj:`bool`): whether or not gradient is used
          - ``bounds`` (:obj:`list`): modified bounds for minimizer
          - ``fixed_vals`` (:obj:`list`): modified fixed values for minimizer

    .. note::

        ``stitch_pars(pars, stitch_with=None)`` is a callable that will
        stitch the fixed parameters of the minimization back into the unfixed
        parameters.

    .. note::

        ``do_stitch`` will modify the ``init_pars``, ``par_bounds``, and ``fixed_vals`` by stripping away the entries associated with fixed parameters. The parameters can be stitched back in via ``stitch_pars``.

    Returns:
        minimizer_kwargs (:obj:`dict`): arguments to pass to a minimizer following the :func:`scipy.optimize.minimize` API (see notes)
        stitch_pars (:obj:`func`): callable that stitches fixed parameters into the unfixed parameters
    """
    tensorlib, _ = get_backend()

    fixed_vals = fixed_vals or []
    fixed_idx = [x[0] for x in fixed_vals]
    fixed_values = [x[1] for x in fixed_vals]
    variable_idx = [x for x in range(pdf.config.npars) if x not in fixed_idx]

    minimizer_kwargs,post_processor = _configure_internal_minimize(init_pars,variable_idx,do_stitch,par_bounds,fixed_idx,fixed_values)



    internal_objective_maybe_grad = _get_internal_objective(
        objective,
        tensorlib.astensor(data),
        pdf,
        post_processor,
        do_grad=do_grad,
        jit_pieces={
            'fixed_idx': fixed_idx,
            'variable_idx': variable_idx,
            'fixed_values': fixed_values,
            'do_stitch': do_stitch,
            'par_bounds': tensorlib.astensor(minimizer_kwargs.pop('variable_bounds'))
        },
    )

    minimizer_kwargs['func'] = internal_objective_maybe_grad
    minimizer_kwargs['do_grad'] = do_grad
    return minimizer_kwargs, post_processor
