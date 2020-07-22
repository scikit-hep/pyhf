"""SciPy Optimizer Class."""
from .mixins import OptimizerMixin
import scipy


class scipy_optimizer(OptimizerMixin):
    """
    Optimizer that uses scipy.optimize.minimize.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the scipy_optimizer.

        See :class:`pyhf.optimize.mixins.OptimizerMixin` for configuration options.
        """
        self.name = 'scipy'
        super(scipy_optimizer, self).__init__(*args, **kwargs)

    def _setup_minimizer(self, objective, pdf, init_pars, par_bounds, fixed_vals=None):
        self._minimizer = scipy.optimize.minimize

    def _minimize(
        self,
        func,
        init,
        method='SLSQP',
        jac=None,
        bounds=None,
        fixed_vals=None,
        options={},
    ):
        """
        Same signature as scipy.optimize.minimize.

        Returns:
            fitresult (`scipy.optimize.OptimizeResult`): the fit result
        """

        fixed_vals = fixed_vals or []
        indices = [i for i, _ in fixed_vals]
        values = [v for _, v in fixed_vals]
        if fixed_vals:
            constraints = [{'type': 'eq', 'fun': lambda v: v[indices] - values}]
        else:
            constraints = []

        return self._minimizer(
            func,
            init,
            method=method,
            jac=jac,
            bounds=bounds,
            constraints=constraints,
            options=dict(maxiter=self.maxiter, disp=self.verbose, **options),
        )
