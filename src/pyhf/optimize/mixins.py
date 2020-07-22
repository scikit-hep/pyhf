"""Helper Classes for use of automatic differentiation."""
from .. import get_backend
from .common import shim

import logging

log = logging.getLogger(__name__)


class OptimizerMixin(object):
    """Mixin Class to build optimizers."""

    def __init__(self, **kwargs):
        """
        Create an optimizer.

        Args:
            maxiter (`int`): maximum number of iterations. Default is 100000.
            verbose (`bool`): print verbose output during minimization. Default is off.
        """
        self.maxiter = kwargs.pop('maxiter', 100000)
        self.verbose = kwargs.pop('verbose', False)
        self._minimizer = None

        if kwargs:
            raise KeyError(
                f"""Unexpected keyword argument(s): '{"', '".join(kwargs.keys())}'"""
            )

    def _internal_minimize(
        self,
        objective,
        init,
        method='SLSQP',
        jac=None,
        bounds=None,
        fixed_vals=None,
        options={},
    ):

        minimizer = self._get_minimizer(objective, init, bounds, fixed_vals=fixed_vals)
        result = self._minimize(
            minimizer,
            objective,
            init,
            method=method,
            jac=jac,
            bounds=bounds,
            fixed_vals=fixed_vals,
            options=options,
        )

        try:
            assert result.success
        except AssertionError:
            log.error(result)
            raise
        return result

    def _internal_postprocess(self, fitresult, stitch_pars):
        """
        Post-process the fit result, ``scipy.optimize.OptimizeResult`` object.

        Returns:
            fitresult (`scipy.optimize.OptimizeResult`): A modified version of the fit result.
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
        return_fit_object=False,
        do_grad=False,
        do_stitch=False,
        method='SLSQP',
        **kwargs,
    ):
        """
        Find parameters that minimize the objective.

        Args:
            objective (`func`): objective function
            data (`list`): observed data
            pdf (`pyhf.pdf.Model`): model
            init_pars (`list`): initial parameters
            par_bounds (`list`): parameter boundaries
            fixed_vals (`list`): fixed parameter values
            return_fitted_val (`bool`): return bestfit value of the objective
            return_fit_object (`bool`): return ``scipy.optimize.OptimizeResult``
            do_grad (`bool`): enable autodifferentiation mode. Default is off.
            do_stitch (`bool`): enable splicing/stitching fixed parameter.
            method (`str`): minimization routine
            kwargs: other options to pass through to underlying minimizer

        Returns:
            parameters (`tensor`): fitted parameters
            minimum (`float`): if ``return_fitted_val`` flagged, return minimized objective value
            result (scipy.optimize.OptimizeResult`): if ``return_fit_object`` flagged
        """
        stitch_pars, wrapped_objective, jac, init, bounds = shim(
            objective,
            data,
            pdf,
            init_pars,
            par_bounds,
            fixed_vals,
            do_grad=do_grad,
            do_stitch=do_stitch,
        )

        minimizer_kwargs = dict(
            method=method, jac=jac, bounds=bounds, fixed_vals=fixed_vals, options=kwargs
        )
        if do_stitch:
            minimizer_kwargs['fixed_vals'] = []

        result = self._internal_minimize(wrapped_objective, init, **minimizer_kwargs)
        result = self._internal_postprocess(result, stitch_pars)

        _returns = [result.x]
        if return_fitted_val:
            _returns.append(result.fun)
        if return_fit_object:
            _returns.append(result)
        return tuple(_returns) if len(_returns) > 1 else _returns[0]
