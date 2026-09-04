"""Optimistix Optimizer Class."""

import optimistix as optx
import scipy

import pyhf
from pyhf.optimize.mixins import OptimizerMixin


class optimistix_optimizer(OptimizerMixin):
    """
    Optimizer that uses :func:`optimistix.minimise`.
    """

    __slots__ = ["name", "options", "solver", "throw"]

    def __init__(self, *args, **kwargs):
        """
        Initialize the optimistix_optimizer.

        See :class:`pyhf.optimize.mixins.OptimizerMixin` for other configuration options.

        Args:
            solver: optimistix.Solver instance.
            options: Runtime args for the solver.
            throw (:obj:`bool`): How to deal with errors. If True, raises exceptions.
        """
        self.name = "optimistix"
        self.solver = kwargs.pop("solver", optx.DFP(rtol=1e-6, atol=1e-6))
        if self.solver is None:
            msg = f"{type(self).__name__} needs a solver"
            raise ValueError(msg)
        self.options = kwargs.pop("options", None)
        self.throw = kwargs.pop("throw", False)

        # the default for maxiter is too high for optimistix, so we'll lower it here:
        self.maxiter = kwargs.pop("maxiter", 1000)
        super().__init__(*args, **kwargs)

    def _get_minimizer(
        self,
        objective_and_grad,  # noqa: ARG002
        init_pars,  # noqa: ARG002
        init_bounds,  # noqa: ARG002
        fixed_vals=None,  # noqa: ARG002
        do_grad=False,  # noqa: ARG002
        par_names=None,  # noqa: ARG002
    ):
        return optx.minimise

    def _minimize(
        self,
        minimizer,
        func,
        x0,
        do_grad=False,  # noqa: ARG002
        bounds=None,  # noqa: ARG002
        fixed_vals=None,
        options=None,
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
        if pyhf.tensorlib.name != "jax":
            msg = (
                "The optimistix optimizer can only be used with the JAX backend. "
                "Use `pyhf.set_backend('jax', pyhf.optimize.optimistix_optimizer(...)) "
                "at the top level of your python code to change your backend to JAX."
            )
            raise ValueError(msg)

        options = options or {}
        max_steps = options.pop("maxiter", self.maxiter)
        solver = options.pop("solver", self.solver)
        throw = options.pop("throw", self.throw)
        options = options.pop("options", self.options)

        fixed_vals = fixed_vals or []
        # for optimistix we re-express the func such that the fixed values will be
        # 'patched' into the correct indices again. This trick essentially zeros the
        # gradients of fixed values because the optimizer never sees them.
        # We also only pass the free params to the optimizer (without the fixed ones!)
        # that way the hessian is only as big as it has to be. This is much more efficient
        # as we only calculate gradients for the free parameters and not just zero the
        # gradients for the fixed ones manually.
        x0_full = pyhf.tensorlib.astensor(x0, dtype="float")

        fixed_idxs, fixed_vals_arr = zip(*fixed_vals) if fixed_vals else ((), ())
        fixed_idxs = pyhf.tensorlib.astensor(fixed_idxs, dtype="int")
        fixed_vals_arr = pyhf.tensorlib.astensor(fixed_vals_arr, dtype="float")

        npars = x0_full.shape[0]
        all_idxs = pyhf.tensorlib.arange(npars)
        free_mask = pyhf.tensorlib.ones(npars, dtype="bool").at[fixed_idxs].set(False)
        free_idxs = all_idxs[free_mask]

        x0_free = x0_full[free_idxs]

        def wrapped_func(x0_free, args):
            free_idxs, fixed_idxs, fixed_vals_arr, npars = args

            x_full = pyhf.tensorlib.empty((npars,), dtype="float")
            x_full = x_full.at[free_idxs].set(x0_free)
            x_full = x_full.at[fixed_idxs].set(fixed_vals_arr)

            return func(x_full)

        args = (free_idxs, fixed_idxs, fixed_vals_arr, npars)

        res = minimizer(
            fn=wrapped_func,
            solver=solver,
            y0=x0_free,
            args=args,
            max_steps=max_steps,
            throw=throw,
            has_aux=True,
            options=options,
        )

        # reconstruct output params vector
        x_res = pyhf.tensorlib.empty((npars,), dtype="float")
        x_res = x_res.at[free_idxs].set(res.value)
        x_res = x_res.at[fixed_idxs].set(fixed_vals_arr)

        converged = res.result == optx.RESULTS.successful
        message = "Optimization terminated successfully."
        if not converged:
            message = "Optimization failed."

        return scipy.optimize.OptimizeResult(
            x=x_res.tolist(),
            unc=None,
            corr=None,
            success=converged,
            fun=res.state.f_info.f,
            hess_inv=None,
            message=message,
            nfev=None,
            nit=res.stats["num_steps"],
            optx_state=res.state,
        )
