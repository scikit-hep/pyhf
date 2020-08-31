"""Helper Classes for use of automatic differentiation."""
from .. import get_backend, exceptions
from .common import shim

import logging

log = logging.getLogger(__name__)


class OptimizerMixin(object):
    """Mixin Class to build optimizers."""

    __slots__ = ['maxiter', 'verbose']

    def __init__(self, **kwargs):
        """
        Create an optimizer.

        Args:
            maxiter (`int`): maximum number of iterations. Default is 100000.
            verbose (`int`): verbose output level during minimization. Default is off (0).
        """
        self.maxiter = kwargs.pop('maxiter', 100000)
        self.verbose = kwargs.pop('verbose', 0)

        if kwargs:
            raise exceptions.Unsupported(
                f"Unsupported kwargs were passed in: {list(kwargs.keys())}."
            )

    def _internal_minimize(
        self, func, x0, do_grad=False, bounds=None, fixed_vals=None, options={}
    ):

        minimizer = self._get_minimizer(
            func, x0, bounds, fixed_vals=fixed_vals, do_grad=do_grad
        )
        result = self._minimize(
            minimizer,
            func,
            x0,
            do_grad=do_grad,
            bounds=bounds,
            fixed_vals=fixed_vals,
            options=options,
        )

        try:
            assert result.success
        except AssertionError:
            log.error(result)
            raise exceptions.FailedMinimization(result)
        return result

    def _internal_postprocess(self, fitresult, stitch_pars):
        """
        Post-process the fit result.

        Returns:
            fitresult (scipy.optimize.OptimizeResult): A modified version of the fit result.
        """
        tensorlib, _ = get_backend()

        fitted_pars = stitch_pars(tensorlib.astensor(fitresult.x))
        # extract number of fixed parameters
        num_fixed_pars = len(fitted_pars) - len(fitresult.x)

        # check if uncertainties were provided
        uncertainties = getattr(fitresult, 'unc', None)
        if uncertainties is not None:
            # stitch in zero-uncertainty for fixed values
            fitted_uncs = stitch_pars(
                tensorlib.astensor(uncertainties),
                stitch_with=tensorlib.zeros(num_fixed_pars),
            )
            fitresult.unc = fitted_uncs
            fitted_pars = tensorlib.stack([fitted_pars, fitted_uncs], axis=1)

        fitresult.x = fitted_pars
        fitresult.fun = tensorlib.astensor(fitresult.fun)

        return fitresult

    def minimize(
        self,
        objective,
        data,
        pdf,
        init_pars,
        par_bounds,
        fixed_vals=None,
        return_fitted_val=False,
        return_result_obj=False,
        do_grad=None,
        do_stitch=False,
        **kwargs,
    ):
        """
        Find parameters that minimize the objective.

        Args:
            objective (`func`): objective function
            data (`list`): observed data
            pdf (~pyhf.pdf.Model): The statistical model adhering to the schema model.json
            init_pars (`list`): initial parameters
            par_bounds (`list`): parameter boundaries
            fixed_vals (`list`): fixed parameter values
            return_fitted_val (`bool`): return bestfit value of the objective
            return_result_obj (`bool`): return :class:`scipy.optimize.OptimizeResult`
            do_grad (`bool`): enable autodifferentiation mode. Default depends on backend (:attr:`pyhf.tensorlib.default_do_grad`).
            do_stitch (`bool`): enable splicing/stitching fixed parameter.
            kwargs: other options to pass through to underlying minimizer

        Returns:
            Fitted parameters or tuple of results:

                - parameters (`tensor`): fitted parameters
                - minimum (`float`): if ``return_fitted_val`` flagged, return minimized objective value
                - result (:class:`scipy.optimize.OptimizeResult`): if ``return_result_obj`` flagged
        """
        # Configure do_grad based on backend "automagically" if not set by user
        tensorlib, _ = get_backend()
        do_grad = tensorlib.default_do_grad if do_grad is None else do_grad

        minimizer_kwargs, stitch_pars = shim(
            objective,
            data,
            pdf,
            init_pars,
            par_bounds,
            fixed_vals,
            do_grad=do_grad,
            do_stitch=do_stitch,
        )

        result = self._internal_minimize(**minimizer_kwargs, options=kwargs)
        result = self._internal_postprocess(result, stitch_pars)

        _returns = [result.x]
        if return_fitted_val:
            _returns.append(result.fun)
        if return_result_obj:
            _returns.append(result)
        return tuple(_returns) if len(_returns) > 1 else _returns[0]
