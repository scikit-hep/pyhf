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
        func,
        init,
        method='SLSQP',
        jac=None,
        bounds=None,
        fixed_vals=None,
        options={},
    ):
        result = self._minimize(
            func,
            init,
            method=method,
            bounds=bounds,
            fixed_vals=fixed_vals,
            options=options,
            jac=jac,
        )

        try:
            assert result.success
        except AssertionError:
            log.error(result)
            raise
        return result

    def minimize(
        self,
        objective,
        data,
        pdf,
        init_pars,
        par_bounds,
        fixed_vals=None,
        return_fitted_val=False,
        do_grad=False,
        do_stitch=False,
        method='SLSQP',
        **kwargs,
    ):
        """
        Find Function Parameters that minimize the Objective.

        Args:
            objective: objective function
            data: observed data
            pdf: model
            init_pars: initial parameters
            par_bounds: parameter boundaries
            fixed_vals: fixed parameter values
            return_fitted_val: return bestfit value of the objective
            do_grad (`bool`): enable autodifferentiation mode. Default is off.
            do_stitch (`bool`): enable splicing/stitching fixed parameter.
            method: minimization routine
            kwargs: other options to pass through to underlying minimizer

        Returns:
            bestfit parameters

        """
        tensorlib, _ = get_backend()
        tv, fixed_values_tensor, func_and_grad, init, bounds = shim(
            objective,
            data,
            pdf,
            init_pars,
            par_bounds,
            fixed_vals,
            do_grad=do_grad,
            do_stitch=do_stitch,
        )
        if do_grad:
            func = lambda pars: func_and_grad(pars)[0]
            jac = lambda pars: func_and_grad(pars)[1]
        else:
            func = func_and_grad
            jac = None

        if do_stitch:
            self._setup_minimizer(func, pdf, init_pars, par_bounds, [])
        else:
            self._setup_minimizer(func, pdf, init_pars, par_bounds, fixed_vals)

        minimizer_kwargs = dict(method=method, bounds=bounds, options=kwargs, jac=jac)
        if not do_stitch:
            minimizer_kwargs.update(dict(fixed_vals=fixed_vals))
        result = self._internal_minimize(func, init, **minimizer_kwargs)

        nonfixed_vals = tensorlib.astensor(result.x)
        fitted_val = result.fun
        # stitch things back up if needed
        if do_stitch:
            fitted_pars = tv.stitch([fixed_values_tensor, nonfixed_vals])
        else:
            fitted_pars = nonfixed_vals

        # check if uncertainties were provided
        uncertainties = getattr(result, 'unc', None)
        if uncertainties is not None:
            if do_stitch:
                # stitch in zero-uncertainty for fixed values
                fitted_uncs = tv.stitch(
                    [
                        tensorlib.zeros(fixed_values_tensor.shape),
                        tensorlib.astensor(uncertainties),
                    ]
                )
            else:
                fitted_uncs = tensorlib.astensor(uncertainties)
            fitted_pars = tensorlib.stack([fitted_pars, fitted_uncs], axis=1)
        if return_fitted_val:
            return fitted_pars, tensorlib.astensor(fitted_val)
        return fitted_pars
