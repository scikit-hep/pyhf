"""SciPy Optimizer Class."""
from pyhf import exceptions
from pyhf.optimize.mixins import OptimizerMixin
import scipy


class scipy_optimizer(OptimizerMixin):
    """
    Optimizer that uses :func:`scipy.optimize.minimize`.
    """

    __slots__ = ['name', 'tolerance', 'solver_options']

    def __init__(self, *args, **kwargs):
        """
        Initialize the scipy_optimizer.

        See :class:`pyhf.optimize.mixins.OptimizerMixin` for other configuration options.

        Args:
            tolerance (:obj:`float`): Tolerance for termination.
              See specific optimizer for detailed meaning.
              Default is ``None``.
            solver_options (:obj:`dict`): additional solver options. See
                :func:`scipy.optimize.show_options` for additional options of
                optimization solvers.
        """
        self.name = 'scipy'
        self.tolerance = kwargs.pop('tolerance', None)
        self.solver_options = kwargs.pop('solver_options', {})
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
        return scipy.optimize.minimize

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
        Same signature as :func:`scipy.optimize.minimize`.

        Minimizer Options:
          * maxiter (:obj:`int`): Maximum number of iterations. Default is ``100000``.
          * verbose (:obj:`bool`): Print verbose output during minimization.
            Default is ``False``.
          * method (:obj:`str`): Minimization routine. Default is ``'SLSQP'``.
          * tolerance (:obj:`float`): Tolerance for termination. See specific optimizer
            for detailed meaning.
            Default is ``None``.
          * solver_options (:obj:`dict`): additional solver options. See
            :func:`scipy.optimize.show_options` for additional options of
            optimization solvers.

        Returns:
            fitresult (scipy.optimize.OptimizeResult): the fit result
        """
        maxiter = options.pop('maxiter', self.maxiter)
        verbose = options.pop('verbose', self.verbose)
        method = options.pop('method', 'SLSQP')
        tolerance = options.pop('tolerance', self.tolerance)
        solver_options = options.pop('solver_options', self.solver_options)
        if options:
            raise exceptions.Unsupported(
                f"Unsupported options were passed in: {list(options.keys())}."
            )

        fixed_vals = fixed_vals or []
        indices = [i for i, _ in fixed_vals]
        values = [v for _, v in fixed_vals]
        if fixed_vals:
            constraints = [{'type': 'eq', 'fun': lambda v: v[indices] - values}]
            # update the initial values to the fixed value for any fixed parameter
            for idx, fixed_val in fixed_vals:
                x0[idx] = fixed_val
        else:
            constraints = []

        return minimizer(
            func,
            x0,
            method=method,
            jac=do_grad,
            bounds=bounds,
            constraints=constraints,
            tol=tolerance,
            options=dict(maxiter=maxiter, disp=bool(verbose), **solver_options),
        )
