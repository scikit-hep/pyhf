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

    def _minimize(
        self,
        objective,
        init,
        method='SLSQP',
        jac=None,
        bounds=None,
        fixed_vals=None,
        options={},
    ):
        """
        Same signature as scipy.optimize.minimize.

        This returns a callable that returns the fitresult.

        Returns:
            fitresult (`scipy.optimize.OptimizeResult`): the fit result
        """
        self._minimizer = scipy.optimize.minimize

        fixed_vals = fixed_vals or []
        indices = [i for i, _ in fixed_vals]
        values = [v for _, v in fixed_vals]
        if fixed_vals:
            constraints = [{'type': 'eq', 'fun': lambda v: v[indices] - values}]
        else:
            constraints = []

        return self._minimizer(
            objective,
            init,
            method=method,
            jac=jac,
            bounds=bounds,
            constraints=constraints,
            options=dict(maxiter=self.maxiter, disp=self.verbose, **options),
        )
