"""Minuit Optimizer Class."""
from .. import default_backend, exceptions
from .mixins import OptimizerMixin
import scipy
import iminuit


class minuit_optimizer(OptimizerMixin):
    """
    Optimizer that uses iminuit.Minuit.migrad.
    """

    __slots__ = ['name', 'errordef', 'steps']

    def __init__(self, *args, **kwargs):
        """
        Create MINUIT Optimizer.

        .. note::

            ``errordef`` should be 1.0 for a least-squares cost function and 0.5
            for negative log-likelihood function. See page 37 of
            http://hep.fi.infn.it/minuit.pdf. This parameter is sometimes
            called ``UP`` in the ``MINUIT`` docs.


        Args:
            errordef (`float`): See minuit docs. Default is 1.0.
            steps (`int`): Number of steps for the bounds. Default is 1000.
        """
        self.name = 'minuit'
        self.errordef = kwargs.pop('errordef', 1)
        self.steps = kwargs.pop('steps', 1000)
        super().__init__(*args, **kwargs)

    def _get_minimizer(
        self, objective_and_grad, init_pars, init_bounds, fixed_vals=None, do_grad=False
    ):

        step_sizes = [(b[1] - b[0]) / float(self.steps) for b in init_bounds]
        fixed_vals = fixed_vals or []
        # Minuit wants True/False for each parameter
        fixed_bools = [False] * len(init_pars)
        for index, val in fixed_vals:
            fixed_bools[index] = True
            init_pars[index] = val
            step_sizes[index] = 0.0

        # Minuit requires jac=callable
        if do_grad:
            wrapped_objective = lambda pars: objective_and_grad(pars)[0]
            jac = lambda pars: objective_and_grad(pars)[1]
        else:
            wrapped_objective = objective_and_grad
            jac = None

        kwargs = dict(
            fcn=wrapped_objective,
            grad=jac,
            start=init_pars,
            error=step_sizes,
            limit=init_bounds,
            fix=fixed_bools,
            print_level=self.verbose,
            errordef=self.errordef,
        )
        return iminuit.Minuit.from_array_func(**kwargs)

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

        Note: an additional `minuit` is injected into the fitresult to get the
        underlying minimizer.

        Minimizer Options:
            maxiter (`int`): maximum number of iterations. Default is 100000.
            return_uncertainties (`bool`): Return uncertainties on the fitted parameters. Default is off.

        Returns:
            fitresult (scipy.optimize.OptimizeResult): the fit result
        """
        maxiter = options.pop('maxiter', self.maxiter)
        return_uncertainties = options.pop('return_uncertainties', False)
        if options:
            raise exceptions.Unsupported(
                f"Unsupported options were passed in: {list(options.keys())}."
            )

        minimizer.migrad(ncall=maxiter)
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

        n = len(x0)
        hess_inv = default_backend.ones((n, n))
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
