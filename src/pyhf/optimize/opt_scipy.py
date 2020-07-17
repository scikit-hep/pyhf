"""SciPy Optimizer Class."""
from .mixins import OptimizerMixin
import scipy


class scipy_optimizer(OptimizerMixin):
    def __init__(self, *args, **kwargs):
        self.name = 'scipy'
        super(scipy_optimizer, self).__init__(*args, **kwargs)

    def _setup_minimizer(
        self, objective, data, pdf, init_pars, par_bounds, fixed_vals=None
    ):
        self._minimizer = scipy.optimize.minimize

    def _minimize(self, func, init, method='SLSQP', bounds=None, options={}, jac=None):
        """
        Same signature as scipy.optimize.minimize.

        Returns:
            fitresult (`scipy.optimize.OptimizeResult`): the fit result
        """
        return self._minimizer(
            func,
            init,
            method=method,
            jac=jac,
            bounds=bounds,
            options=dict(maxiter=self.maxiter, disp=self.verbose, **options),
        )
