"""Minuit Optimizer Class."""
from .mixins import OptimizerMixin
import scipy
import numpy as np
import iminuit


class minuit_optimizer(OptimizerMixin):
    """
    Optimizer that uses iminuit.Minuit.migrad.
    """

    def __init__(self, *args, **kwargs):
        """
        Create MINUIT Optimizer.

        .. note::

            errordef should be 1.0 for a least-squares cost function and 0.5
            for negative log-likelihood function. See page 37 of
            http://hep.fi.infn.it/minuit.pdf. This parameter is sometimes
            called UP in the MINUIT docs.


        Args:
            errordef (`float`): See minuit docs. Default is 1.0.
            steps (`int`): Number of steps for the bounds. Default is 1000.
        """
        self.errordef = kwargs.pop('errordef', 1)
        self.steps = kwargs.pop('steps', 1000)
        self.name = 'minuit'
        super(minuit_optimizer, self).__init__(*args, **kwargs)

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
            errordef=self.errordef,
            use_array_call=True,
            name=parnames,
            **kwargs,
        )

    def _minimize(self, func, init, method='SLSQP', jac=None, bounds=None, options={}):
        """
        Same signature as scipy.optimize.minimize.

        Note: an additional `minuit` is injected into the fitresult to get the
        underlying minimizer.

        Minimizer Options:
            return_uncertainties (`bool`): Return uncertainties on the fitted parameters. Default is off.

        Returns:
            fitresult (`scipy.optimize.OptimizeResult`): the fit result
        """
        return_uncertainties = options.pop('return_uncertainties', False)
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
        hess_inv = np.ones((n, n))
        if self._minimizer.valid:
            hess_inv = self._minimizer.np_covariance()

        unc = None
        if return_uncertainties:
            unc = self._minimizer.np_errors()

        return scipy.optimize.OptimizeResult(
            x=self._minimizer.np_values(),
            unc=unc,
            success=self._minimizer.valid,
            fun=self._minimizer.fval,
            hess_inv=hess_inv,
            message=message,
            nfev=self._minimizer.ncalls,
            njev=self._minimizer.ngrads,
            minuit=self._minimizer,
        )
