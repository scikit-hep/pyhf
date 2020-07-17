"""Helper Classes for use of automatic differentiation."""
import scipy
import iminuit
import logging
import numpy as np

log = logging.getLogger(__name__)


class OptimizerMixin(object):
    """Mixin Class to build optimizers."""

    def __init__(self, **kwargs):
        """Create Mixin for optimizers."""
        self.maxiter = kwargs.pop('maxiter', 100000)
        self.verbose = kwargs.pop('verbose', False)
        self.grad = kwargs.pop('grad', False)
        self.minimizer = None

        if kwargs:
            raise KeyError(
                f"""Unexpected keyword argument(s): '{"', '".join(kwargs.keys())}'"""
            )

    def minimize(
        self, func, init, method='SLSQP', jac=None, bounds=None, options={},
    ):
        """
        Find Function Parameters that minimize the Objective.

        Returns:
            bestfit parameters

        """
        assert self.minimizer, "Minimizer must be setup before being used."
        if self.grad:
            jac = None
        result = self._minimize(
            func, init, method=method, bounds=bounds, options=options, jac=jac
        )
        try:
            assert result.success
        except AssertionError:
            log.error(result)
            raise
        return result


class ScipyOptimizer(OptimizerMixin):
    def __init__(self, *args, **kwargs):
        self.minimizer_name = 'scipy'
        super(ScipyOptimizer, self).__init__(*args, **kwargs)

    def _setup_minimizer(
        self, objective, data, pdf, init_pars, par_bounds, fixed_vals=None
    ):
        self.minimizer = scipy.optimize.minimize

    def _minimize(self, func, init, method='SLSQP', bounds=None, options={}, jac=None):
        return self.minimizer(
            func,
            init,
            method=method,
            jac=jac,
            bounds=bounds,
            options=dict(maxiter=self.maxiter, disp=self.verbose, **options),
        )


class MinuitOptimizer(OptimizerMixin):
    def __init__(self, *args, **kwargs):
        """
        Create MINUIT Optimizer.

        Args:
            verbose (`bool`): print verbose output during minimization

        """
        self.errordef = kwargs.get('errordef', 1)
        self.steps = kwargs.get('steps', 1000)
        self.minimizer_name = 'minuit'
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
        self.minimizer = iminuit.Minuit(
            f,
            print_level=1 if self.verbose else 0,
            errordef=1,
            use_array_call=True,
            forced_parameters=parnames,
            **kwargs,
        )

    def _minimize(self, func, init, method='SLSQP', jac=None, bounds=None, options={}):
        self.minimizer.migrad(ncall=self.maxiter)
        # Following lines below come from:
        # https://github.com/scikit-hep/iminuit/blob/22f6ed7146c1d1f3274309656d8c04461dde5ba3/src/iminuit/_minimize.py#L106-L125
        message = "Optimization terminated successfully."
        if not self.minimizer.valid:
            message = "Optimization failed."
            fmin = self.minimizer.fmin
            if fmin.has_reached_call_limit:
                message += " Call limit was reached."
            if fmin.is_above_max_edm:
                message += " Estimated distance to minimum too large."

        n = len(init)
        return scipy.optimize.OptimizeResult(
            x=self.minimizer.np_values(),
            success=self.minimizer.valid,
            fun=self.minimizer.fval,
            hess_inv=self.minimizer.np_covariance()
            if self.minimizer.valid
            else np.ones((n, n)),
            message=message,
            nfev=self.minimizer.ncalls,
            njev=self.minimizer.ngrads,
            minuit=self.minimizer,
        )
