"""Helper Classes for use of automatic differentiation."""
from .. import get_backend
from .common import shim

import logging

log = logging.getLogger(__name__)


class OptimizerMixin(object):
    """Mixin Class to build optimizers."""

    def __init__(self, **kwargs):
        """Create Mixin for optimizers."""
        self.maxiter = kwargs.pop('maxiter', 100000)
        self.verbose = kwargs.pop('verbose', False)
        self.grad = kwargs.pop('grad', False)
        self._minimizer = None

        if kwargs:
            raise KeyError(
                f"""Unexpected keyword argument(s): '{"', '".join(kwargs.keys())}'"""
            )

    def _internal_minimize(
        self, func, init, method='SLSQP', jac=None, bounds=None, options={}
    ):
        result = self._minimize(
            func, init, method=method, bounds=bounds, options=options, jac=jac
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
        method='SLSQP',
        minimizer_options={},
    ):
        """
        Find Function Parameters that minimize the Objective.

        Returns:
            bestfit parameters

        """
        self._setup_minimizer(objective, data, pdf, init_pars, par_bounds, fixed_vals)

        tensorlib, _ = get_backend()
        tv, fixed_values_tensor, func_and_grad, init, bounds = shim(
            objective, data, pdf, init_pars, par_bounds, fixed_vals, do_grad=self.grad
        )
        if self.grad:
            func = lambda pars: func_and_grad(pars)[0]
            jac = lambda pars: func_and_grad(pars)[1]
        else:
            func = func_and_grad
            jac = None
        result = self._internal_minimize(
            func, init, method=method, bounds=bounds, options=minimizer_options, jac=jac
        )

        nonfixed_vals = result.x
        fitted_val = result.fun
        fitted_pars = tv.stitch(
            [fixed_values_tensor, tensorlib.astensor(nonfixed_vals)]
        )
        if return_fitted_val:
            return fitted_pars, tensorlib.astensor(fitted_val)
        return fitted_pars
