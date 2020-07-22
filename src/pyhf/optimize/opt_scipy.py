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

    def _get_minimizer(self, objective, init_pars, par_bounds, fixed_vals=None):
        return scipy.optimize.minimize

    def _minimize(
        self,
        minimizer,
        func,
        x0,
        method='SLSQP',
        jac=None,
        bounds=None,
        fixed_vals=None,
        return_uncertainties=False,
        options={},
    ):
        """
        Same signature as scipy.optimize.minimize.

        Returns:
            fitresult (`scipy.optimize.OptimizeResult`): the fit result
        """
        assert (
            'return_uncertainties' not in options
        ), "Optimizer does not support returning uncertainties."

        fixed_vals = fixed_vals or []
        indices = [i for i, _ in fixed_vals]
        values = [v for _, v in fixed_vals]
        if fixed_vals:
            constraints = [{'type': 'eq', 'fun': lambda v: v[indices] - values}]
        else:
            constraints = []

        return minimizer(
            func,
            x0,
            method=method,
            jac=jac,
            bounds=bounds,
            constraints=constraints,
            options=dict(maxiter=self.maxiter, disp=self.verbose, **options),
        )
