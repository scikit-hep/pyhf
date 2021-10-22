"""Common Backend Shim to prepare minimization for optimizer."""
from pyhf.tensor.manager import get_backend
from pyhf.tensor.common import _TensorViewer


def _make_stitch_pars(tv=None, fixed_values=None):
    """
    Construct a callable to stitch fixed parameter values into the unfixed parameters. See :func:`shim`.

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

    def stitch_pars(pars, stitch_with=fixed_values):
        tb, _ = get_backend()
        return tv.stitch([tb.astensor(stitch_with, dtype='float'), pars])

    return stitch_pars


def _get_tensor_shim():
    """
    A shim-retriever to lazy-retrieve the necessary shims as needed.

    Because pyhf.tensor is a lazy-retriever for the backends, we can be sure
    that tensorlib is imported correctly.
    """
    tensorlib, _ = get_backend()
    if tensorlib.name == 'numpy':
        from pyhf.optimize.opt_numpy import wrap_objective as numpy_shim

        return numpy_shim

    if tensorlib.name == 'tensorflow':
        from pyhf.optimize.opt_tflow import wrap_objective as tflow_shim

        return tflow_shim

    if tensorlib.name == 'pytorch':
        from pyhf.optimize.opt_pytorch import wrap_objective as pytorch_shim

        return pytorch_shim

    if tensorlib.name == 'jax':
        from pyhf.optimize.opt_jax import wrap_objective as jax_shim

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
        objective (:obj:`func`): objective function
        data (:obj:`list`): observed data
        pdf (~pyhf.pdf.Model): The statistical model adhering to the schema model.json
        init_pars (:obj:`list` of :obj:`float`): The starting values of the model parameters for minimization.
        par_bounds (:obj:`list` of :obj:`list`/:obj:`tuple`): The extrema of values the model parameters
            are allowed to reach in the fit.
            The shape should be ``(n, 2)`` for ``n`` model parameters.
        fixed_vals (:obj:`list` of :obj:`list`/:obj:`tuple`): The pairs of index and constant value for a constant
            model parameter during minimization. Set to ``None`` to allow all parameters to float.

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

    if do_stitch:
        all_init = tensorlib.astensor(init_pars)
        variable_init = tensorlib.tolist(
            tensorlib.gather(all_init, tensorlib.astensor(variable_idx, dtype='int'))
        )
        variable_bounds = [par_bounds[i] for i in variable_idx]
        # stitched out the fixed values, so we don't pass any to the underlying minimizer
        minimizer_fixed_vals = []

        tv = _TensorViewer([fixed_idx, variable_idx])
        # NB: this is a closure, tensorlib needs to be accessed at a different point in time
        stitch_pars = _make_stitch_pars(tv, fixed_values)

    else:
        variable_init = init_pars
        variable_bounds = par_bounds
        minimizer_fixed_vals = fixed_vals
        stitch_pars = _make_stitch_pars()

    objective_and_grad = _get_tensor_shim()(
        objective,
        tensorlib.astensor(data),
        pdf,
        stitch_pars,
        do_grad=do_grad,
        jit_pieces={
            'fixed_idx': fixed_idx,
            'variable_idx': variable_idx,
            'fixed_values': fixed_values,
            'do_stitch': do_stitch,
        },
    )

    minimizer_kwargs = dict(
        func=objective_and_grad,
        x0=variable_init,
        do_grad=do_grad,
        bounds=variable_bounds,
        fixed_vals=minimizer_fixed_vals,
    )

    return minimizer_kwargs, stitch_pars
