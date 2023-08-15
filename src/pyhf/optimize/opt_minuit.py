"""Minuit Optimizer Class."""
from pyhf import exceptions
from pyhf.optimize.mixins import OptimizerMixin
import scipy
import iminuit


class minuit_optimizer(OptimizerMixin):
    """
    Optimizer that minimizes via :meth:`iminuit.Minuit.migrad`.
    """

    __slots__ = ['name', 'errordef', 'steps', 'strategy', 'tolerance']

    def __init__(self, *args, **kwargs):
        """
        Create :class:`iminuit.Minuit` optimizer.

        .. note::

            ``errordef`` should be 1.0 for a least-squares cost function and 0.50
            for negative log-likelihood function --- see `MINUIT: Function Minimization
            and Error Analysis Reference Manual <https://cdsweb.cern.ch/record/2296388/>`_
            Section 7.1: Function normalization and ERROR DEF.
            This parameter is sometimes called ``UP`` in the ``MINUIT`` docs.


        Args:
            errordef (:obj:`float`): See minuit docs. Default is ``1.0``.
            steps (:obj:`int`): Number of steps for the bounds. Default is ``1000``.
            strategy (:obj:`int`): See :attr:`iminuit.Minuit.strategy`.
              Default is ``None``, which results in either
              :attr:`iminuit.Minuit.strategy` ``0`` or ``1`` from the evaluation of
              ``int(not pyhf.tensorlib.default_do_grad)``.
            tolerance (:obj:`float`): Tolerance for termination.
              See specific optimizer for detailed meaning.
              Default is ``0.1``.
        """
        self.name = 'minuit'
        self.errordef = kwargs.pop('errordef', 1)
        self.steps = kwargs.pop('steps', 1000)
        self.strategy = kwargs.pop('strategy', None)
        self.tolerance = kwargs.pop('tolerance', 0.1)
        super().__init__(*args, **kwargs)

    def _get_minimizer(
        self,
        objective_and_grad,
        init_pars,
        init_bounds,
        fixed_vals=None,
        do_grad=False,
        par_names=None,
    ):
        fixed_vals = fixed_vals or []
        # Minuit wants True/False for each parameter
        fixed_bools = [False] * len(init_pars)
        for index, val in fixed_vals:
            fixed_bools[index] = True
            init_pars[index] = val

        # Minuit requires jac=callable
        if do_grad:
            wrapped_objective = lambda pars: objective_and_grad(pars)[0]  # noqa: E731
            jac = lambda pars: objective_and_grad(pars)[1]  # noqa: E731
        else:
            wrapped_objective = objective_and_grad
            jac = None

        minuit = iminuit.Minuit(wrapped_objective, init_pars, grad=jac, name=par_names)
        minuit.limits = init_bounds
        minuit.fixed = fixed_bools
        minuit.print_level = self.verbose
        minuit.errordef = self.errordef
        return minuit

    def _minimize(
        self,
        minimizer,
        func,
        x0,
        do_grad=False,
        bounds=None,
        fixed_vals=None,
        options={},
    ):
        """
        Same signature as :func:`scipy.optimize.minimize`.

        Note: an additional `minuit` is injected into the fitresult to get the
        underlying minimizer.

        Minimizer Options:
          * maxiter (:obj:`int`): Maximum number of iterations. Default is ``100000``.
          * strategy (:obj:`int`): See :attr:`iminuit.Minuit.strategy`.
            Default is to configure in response to ``do_grad``.
          * tolerance (:obj:`float`): Tolerance for termination.
            See specific optimizer for detailed meaning.
            Default is ``0.1``.

        Returns:
            fitresult (scipy.optimize.OptimizeResult): the fit result
        """
        maxiter = options.pop('maxiter', self.maxiter)
        # do_grad value results in iminuit.Minuit.strategy of either:
        #   0: Fast. Does not check a user-provided gradient.
        #   1: Default. Checks user-provided gradient against numerical gradient.
        strategy = options.pop("strategy", self.strategy)
        # Guard against None from either self.strategy defaulting to None or
        # passing strategy=None as options kwarg
        if strategy is None:
            strategy = 0 if do_grad else 1
        tolerance = options.pop('tolerance', self.tolerance)
        if options:
            raise exceptions.Unsupported(
                f"Unsupported options were passed in: {list(options.keys())}."
            )

        minimizer.strategy = strategy
        minimizer.tol = tolerance
        minimizer.migrad(ncall=maxiter)
        # Following lines below come from:
        # https://github.com/scikit-hep/iminuit/blob/23bad7697e39d363f259ca8349684df939b1b2e6/src/iminuit/_minimize.py#L111-L130
        message = "Optimization terminated successfully."
        if not minimizer.valid:
            message = "Optimization failed."
            fmin = minimizer.fmin
            if fmin.has_reached_call_limit:
                message += " Call limit was reached."
            if fmin.is_above_max_edm:
                message += " Estimated distance to minimum too large."

        hess_inv = None
        corr = None
        unc = None
        if minimizer.valid:
            # Extra call to hesse() after migrad() is always needed for good error estimates. If you pass a user-provided gradient to MINUIT, convergence is faster.
            minimizer.hesse()
            hess_inv = minimizer.covariance
            corr = hess_inv.correlation()
            unc = minimizer.errors

        return scipy.optimize.OptimizeResult(
            x=minimizer.values,
            unc=unc,
            corr=corr,
            success=minimizer.valid,
            fun=minimizer.fval,
            hess_inv=hess_inv,
            message=message,
            nfev=minimizer.nfcn,
            njev=minimizer.ngrad,
            minuit=minimizer,
        )
