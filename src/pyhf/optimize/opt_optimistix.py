"""Optimistix Optimizer Class."""

import pyhf
from pyhf.optimize.mixins import OptimizerMixin
import scipy
import optimistix as optx


class optimistix_optimizer(OptimizerMixin):
    """
    Optimizer that uses :func:`scipy.optimize.minimize`.
    """

    __slots__ = ['name', 'options', 'solver', 'throw']

    def __init__(self, *args, **kwargs):
        """
        Initialize the optimistix_optimizer.

        See :class:`pyhf.optimize.mixins.OptimizerMixin` for other configuration options.

        Args:
            solver: optimistix.Solver instance.
            options: Runtime args for the solver.
            throw (:obj:`bool`): How to deal with errors. If True, raises exceptions.
        """
        self.name = 'optimistix'
        self.solver = kwargs.pop('solver', optx.DFP(rtol=1e-6, atol=1e-6))
        if self.solver is None:
            raise ValueError(f'{type(self).__name__} needs a solver')
        self.options = kwargs.pop('options', None)
        self.throw = kwargs.pop('throw', True)

        # the default for maxiter is too high for optimistix, so we'll lower it here:
        self.maxiter = kwargs.pop('maxiter', 1000)
        super().__init__(*args, **kwargs)

    def _get_minimizer(
        self,
        objective_and_grad,
        init_pars,
        init_bounds,
        fixed_vals=None,
        do_grad=False,
        par_names=None,
    ):
        return optx.minimise

    def _minimize(
        self,
        minimizer,
        func,
        x0,
        do_grad=False,
        bounds=None,
        fixed_vals=None,
        options={},
    ):
        """
        Similar signature as :func:`optimistix.minimise`.

        Minimizer Options:
          * maxiter (:obj:`int`): Maximum number of iterations. Default is ``1000``.
          * verbose (:obj:`bool`): Print verbose output during minimization. Unused.
          * solver: optimistix.Solver instance.
          * options: Runtime args for the solver.
          * throw (:obj:`bool`): How to deal with errors. If True, raises exceptions.

        Returns:
            fitresult (scipy.optimize.OptimizeResult): the fit result
        """
        # first make sure we're working with the JAX backend,
        # otherwise we can't minimise with optimistix
        if not pyhf.tensorlib.name == 'jax':
            raise ValueError(
                "The optimistix optimizer can only be used with the JAX backend. " \
                "Use `pyhf.set_backend('jax', pyhf.optimize.optimistix_optimizer(...)) " \
                "at the top level of your python code to change your backend to JAX."
            )

        max_steps = options.pop('maxiter', self.maxiter)
        solver = options.pop('solver', self.solver)
        throw = options.pop('throw', self.throw)
        options = options.pop('options', self.options)

        fixed_vals = fixed_vals or []
        # for optimistix we re-express the func such that the x0 only contains
        # floating point values for params (x0) that the optimizer should minimise.
        # Fixed parameters are set to None (static pytree leaf-type) and thus won't receive
        # a gradient update by the minimizer. We patch/set back their fixed values
        # in the `wrapped_func` before forwarding them to the original `func`.
        x0_floating = list(x0)
        for idx, _ in fixed_vals:
           x0_floating[idx] = None

        def wrapped_func(x0, args):
            (fixed_vals,) = args
            for idx, val in fixed_vals:
                x0[idx] = val
            return func(x0)

        res = minimizer(
            fn=wrapped_func,
            solver=solver,
            y0=x0_floating,
            args=(fixed_vals,),
            max_steps=max_steps,
            throw=throw,
            has_aux=True,
            options=options,
        )

        # patch fixed values back in:
        x = list(res.value)
        for idx, val in fixed_vals:
            x[idx] = val

        converged = res.result == optx.RESULTS.successful
        message = "Optimization terminated successfully."
        if not converged:
            message = "Optimization failed."

        return scipy.optimize.OptimizeResult(
            x=x,
            unc=None,
            corr=None,
            success=converged,
            fun=res.state.f_info.f,
            hess_inv=None,
            message=message,
            nfev=None,
            nit=res.stats['num_steps'],
            optx_state=res.state,
        )
