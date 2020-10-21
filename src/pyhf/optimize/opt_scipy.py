"""SciPy Optimizer Class."""
from .. import exceptions
from .mixins import OptimizerMixin
import scipy


class scipy_optimizer(OptimizerMixin):
    """
    Optimizer that uses :func:`scipy.optimize.minimize`.
    """

    __slots__ = ['name']

    def __init__(self, *args, **kwargs):
        """
        Initialize the scipy_optimizer.

        See :class:`pyhf.optimize.mixins.OptimizerMixin` for configuration options.
        """
        self.name = 'scipy'
        super().__init__(*args, **kwargs)

    def _get_minimizer(
        self, objective_and_grad, init_pars, init_bounds, fixed_vals=None, do_grad=False
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
        return_uncertainties=False,
        options={},
    ):
        """
        Same signature as :func:`scipy.optimize.minimize`.

        Minimizer Options:
            maxiter (`int`): maximum number of iterations. Default is 100000.
            verbose (`bool`): print verbose output during minimization. Default is off.
            method (`str`): minimization routine. Default is 'SLSQP'.

        Returns:
            fitresult (scipy.optimize.OptimizeResult): the fit result
        """
        maxiter = options.pop('maxiter', self.maxiter)
        verbose = options.pop('verbose', self.verbose)
        method = options.pop('method', 'SLSQP')
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
            options=dict(maxiter=maxiter, disp=bool(verbose)),
        )
