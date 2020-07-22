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

    def _get_minimizer(self, objective, init_pars, init_bounds, fixed_vals=None):

        parnames = [f'p{i}' for i in range(len(init_pars))]
        kw = {f'limit_p{i}': b for i, b in enumerate(init_bounds)}
        initvals = {f'p{i}': v for i, v in enumerate(init_pars)}
        step_sizes = {
            'error_p{i}': (b[1] - b[0]) / float(self.steps)
            for i, b in enumerate(init_bounds)
        }
        fixed_vals = fixed_vals or []
        constraints = {}
        for index, value in fixed_vals:
            constraints[f'fix_p{index}'] = True
            initvals[f'p{index}'] = value

        return iminuit.Minuit(
            objective,
            print_level=1 if self.verbose else 0,
            errordef=self.errordef,
            use_array_call=True,
            name=parnames,
            **kw,
            **constraints,
            **initvals,
            **step_sizes,
        )

    def _minimize(
        self,
        minimizer,
        objective,
        init,
        method='SLSQP',
        jac=None,
        bounds=None,
        fixed_vals=None,
        return_uncertainties=False,
        options={},
    ):

        """
        Same signature as scipy.optimize.minimize.

        Note: an additional `minuit` is injected into the fitresult to get the
        underlying minimizer.

        Minimizer Options:
            return_uncertainties (`bool`): Return uncertainties on the fitted parameters. Default is off.

        Returns:
            fitresult (`scipy.optimize.OptimizeResult`): the fit result
        """
        assert method == 'SLSQP', "Optimizer only supports 'SLSQP' minimization."
        return_uncertainties = options.pop('return_uncertainties', False)
        minimizer.migrad(ncall=self.maxiter)
        # Following lines below come from:
        # https://github.com/scikit-hep/iminuit/blob/22f6ed7146c1d1f3274309656d8c04461dde5ba3/src/iminuit/_minimize.py#L106-L125
        message = "Optimization terminated successfully."
        if not minimizer.valid:
            message = "Optimization failed."
            fmin = minimizer.fmin
            if fmin.has_reached_call_limit:
                message += " Call limit was reached."
            if fmin.is_above_max_edm:
                message += " Estimated distance to minimum too large."

        n = len(init)
        hess_inv = np.ones((n, n))
        if minimizer.valid:
            hess_inv = minimizer.np_covariance()

        unc = None
        if return_uncertainties:
            unc = minimizer.np_errors()

        return scipy.optimize.OptimizeResult(
            x=minimizer.np_values(),
            unc=unc,
            success=minimizer.valid,
            fun=minimizer.fval,
            hess_inv=hess_inv,
            message=message,
            nfev=minimizer.ncalls,
            njev=minimizer.ngrads,
            minuit=minimizer,
        )
