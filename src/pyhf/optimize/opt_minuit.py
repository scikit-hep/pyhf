"""Minuit Optimizer Class."""
from .mixins import OptimizerMixin
import scipy
import numpy as np
import iminuit


class MinuitOptimizer(OptimizerMixin):
    def __init__(self, *args, **kwargs):
        """
        Create MINUIT Optimizer.

        Args:
            verbose (`bool`): print verbose output during minimization

        """
        self.errordef = kwargs.get('errordef', 1)
        self.steps = kwargs.get('steps', 1000)
        self.name = 'minuit'
        super(MinuitOptimizer, self).__init__(*args, **kwargs)

    def _setup_minimizer(
        self, objective, data, pdf, init_pars, init_bounds, fixed_vals=None
    ):
        def f(pars):
            result = objective(pars, data, pdf)
            logpdf = result[0]
            return logpdf

        parnames = ['p{}'.format(i) for i in range(len(init_pars))]
        kw = {'limit_p{}'.format(i): b for i, b in enumerate(init_bounds)}
        initvals = {'p{}'.format(i): v for i, v in enumerate(init_pars)}
        step_sizes = {
            'error_p{}'.format(i): (b[1] - b[0]) / float(self.steps)
            for i, b in enumerate(init_bounds)
        }
        fixed_vals = fixed_vals or []
        constraints = {}
        for index, value in fixed_vals:
            constraints = {'fix_p{}'.format(index): True}
            initvals['p{}'.format(index)] = value
        kwargs = {}
        for d in [kw, constraints, initvals, step_sizes]:
            kwargs.update(**d)
        self._minimizer = iminuit.Minuit(
            f,
            print_level=1 if self.verbose else 0,
            errordef=1,
            use_array_call=True,
            forced_parameters=parnames,
            **kwargs,
        )

    def _minimize(self, func, init, method='SLSQP', jac=None, bounds=None, options={}):
        """
        Same signature as scipy.optimize.minimize.

        Note: an additional `minuit` is injected into the fitresult to get the
        underlying minimizer.

        Returns:
            fitresult (`scipy.optimize.OptimizeResult`): the fit result
        """
        self._minimizer.migrad(ncall=self.maxiter)
        # Following lines below come from:
        # https://github.com/scikit-hep/iminuit/blob/22f6ed7146c1d1f3274309656d8c04461dde5ba3/src/iminuit/_minimize.py#L106-L125
        message = "Optimization terminated successfully."
        if not self._minimizer.valid:
            message = "Optimization failed."
            fmin = self._minimizer.fmin
            if fmin.has_reached_call_limit:
                message += " Call limit was reached."
            if fmin.is_above_max_edm:
                message += " Estimated distance to minimum too large."

        n = len(init)
        return scipy.optimize.OptimizeResult(
            x=self._minimizer.np_values(),
            success=self._minimizer.valid,
            fun=self._minimizer.fval,
            hess_inv=self._minimizer.np_covariance()
            if self._minimizer.valid
            else np.ones((n, n)),
            message=message,
            nfev=self._minimizer.ncalls,
            njev=self._minimizer.ngrads,
            minuit=self._minimizer,
        )
