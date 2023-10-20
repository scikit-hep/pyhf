"""Helper Classes for use of automatic differentiation."""
import logging

import numpy as np

from pyhf import exceptions
from pyhf.optimize.common import shim
from pyhf.tensor.manager import get_backend


log = logging.getLogger(__name__)

__all__ = ("OptimizerMixin",)


class OptimizerMixin:
    """Mixin Class to build optimizers."""

    __slots__ = ['maxiter', 'verbose']

    def __init__(self, **kwargs):
        """
        Create an optimizer.

        Args:
            maxiter (:obj:`int`): maximum number of iterations. Default is 100000.
            verbose (:obj:`int`): verbose output level during minimization. Default is off (0).
        """
        self.maxiter = kwargs.pop('maxiter', 100000)
        self.verbose = kwargs.pop('verbose', 0)

        if kwargs:
            raise exceptions.Unsupported(
                f"Unsupported kwargs were passed in: {list(kwargs)}."
            )

    def _internal_minimize(
        self,
        func,
        x0,
        do_grad=False,
        bounds=None,
        fixed_vals=None,
        options={},
        par_names=None,
    ):
        minimizer = self._get_minimizer(
            func,
            x0,
            bounds,
            fixed_vals=fixed_vals,
            do_grad=do_grad,
            par_names=par_names,
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
            log.error(result, exc_info=True)
            raise exceptions.FailedMinimization(result)
        return result

    def _internal_postprocess(
        self,
        fitresult,
        stitch_pars,
        using_minuit,
        return_uncertainties=False,
        uncertainties=None,
        hess_inv=None,
        calc_correlations=False,
        fixed_vals=None,
        init_pars=None,
    ):
        """
        Post-process the fit result.

        Returns:
            fitresult (scipy.optimize.OptimizeResult): A modified version of the fit result.
        """
        tensorlib, _ = get_backend()

        # stitch in missing parameters (e.g. fixed parameters)
        fitted_pars = stitch_pars(tensorlib.astensor(fitresult.x))

        # check if uncertainties were provided (and stitch just in case)
        uncertainties = getattr(fitresult, 'unc', None) or uncertainties
        if uncertainties is not None:
            # extract number of fixed parameters
            num_fixed_pars = len(fitted_pars) - len(fitresult.x)

            # Set uncertainties for fixed parameters to 0 manually
            if fixed_vals is not None:  # check for fixed vals
                if using_minuit:
                    # See related discussion here:
                    # https://github.com/scikit-hep/iminuit/issues/762
                    # https://github.com/scikit-hep/pyhf/issues/1918
                    # https://github.com/scikit-hep/cabinetry/pull/346
                    uncertainties = np.where(fitresult.minuit.fixed, 0.0, uncertainties)
                else:
                    # Not using minuit, so don't have `fitresult.minuit.fixed` here: do it manually
                    fixed_bools = [False] * len(init_pars)
                    for index, _ in fixed_vals:
                        fixed_bools[index] = True
                    uncertainties = tensorlib.where(
                        tensorlib.astensor(fixed_bools, dtype="bool"),
                        tensorlib.astensor(0.0),
                        uncertainties,
                    )
            # stitch in zero-uncertainty for fixed values
            uncertainties = stitch_pars(
                tensorlib.astensor(uncertainties),
                stitch_with=tensorlib.zeros(num_fixed_pars),
            )
            if return_uncertainties:
                fitted_pars = tensorlib.stack([fitted_pars, uncertainties], axis=1)

        cov = getattr(fitresult, 'hess_inv', None)
        if cov is None and hess_inv is not None:
            cov = hess_inv

        # we also need to edit the covariance matrix to zero-out uncertainties!
        # NOTE: minuit already does this (https://github.com/scikit-hep/iminuit/issues/762#issuecomment-1207436406)
        if fixed_vals is not None and not using_minuit:
            fixed_bools = [False] * len(init_pars)
            # Convert fixed_bools to a numpy array and reshape to make it a column vector
            fixed_mask = tensorlib.reshape(
                tensorlib.astensor(fixed_bools, dtype="bool"), (-1, 1)
            )
            # Create 2D masks for rows and columns
            row_mask = fixed_mask
            col_mask = tensorlib.transpose(fixed_mask)

            # Use logical OR to combine the masks
            final_mask = row_mask | col_mask

            # Use np.where to set elements of the covariance matrix to 0 where the mask is True
            cov = tensorlib.where(
                final_mask, tensorlib.astensor(0.0), tensorlib.astensor(cov)
            )

        correlations_from_fit = getattr(fitresult, 'corr', None)
        if correlations_from_fit is None and calc_correlations:
            correlations_from_fit = cov / tensorlib.outer(uncertainties, uncertainties)
            correlations_from_fit = tensorlib.where(
                tensorlib.isfinite(correlations_from_fit),
                correlations_from_fit,
                tensorlib.astensor(0.0),
            )

        if correlations_from_fit is not None and not using_minuit:
            _zeros = tensorlib.zeros(num_fixed_pars)
            # possibly a more elegant way to do this
            stitched_columns = [
                stitch_pars(tensorlib.astensor(column), stitch_with=_zeros)
                for column in zip(*correlations_from_fit)
            ]
            stitched_rows = [
                stitch_pars(tensorlib.astensor(row), stitch_with=_zeros)
                for row in zip(*stitched_columns)
            ]
            correlations_from_fit = tensorlib.stack(stitched_rows, axis=1)

        fitresult.x = fitted_pars
        fitresult.fun = tensorlib.astensor(fitresult.fun)
        fitresult.unc = uncertainties
        fitresult.hess_inv = cov
        fitresult.corr = correlations_from_fit

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
        return_uncertainties=False,
        return_correlations=False,
        do_grad=None,
        do_stitch=False,
        **kwargs,
    ):
        """
        Find parameters that minimize the objective.

        Args:
            objective (:obj:`func`): objective function
            data (:obj:`list`): observed data
            pdf (~pyhf.pdf.Model): The statistical model adhering to the schema model.json
            init_pars (:obj:`list` of :obj:`float`): The starting values of the model parameters for minimization.
            par_bounds (:obj:`list` of :obj:`list`/:obj:`tuple`): The extrema of values the model parameters
                are allowed to reach in the fit.
                The shape should be ``(n, 2)`` for ``n`` model parameters.
            fixed_vals (:obj:`list` of :obj:`list`/:obj:`tuple`): The pairs of index and constant value for a constant
                model parameter during minimization. Set to ``None`` to allow all parameters to float.
            return_fitted_val (:obj:`bool`): Return bestfit value of the objective. Default is off (``False``).
            return_result_obj (:obj:`bool`): Return :class:`scipy.optimize.OptimizeResult`. Default is off (``False``).
            return_uncertainties (:obj:`bool`): Return uncertainties on the fitted parameters. Default is off (``False``).
            return_correlations (:obj:`bool`): Return correlations of the fitted parameters. Default is off (``False``).
            do_grad (:obj:`bool`): enable autodifferentiation mode. Default depends on backend (:attr:`pyhf.tensorlib.default_do_grad`).
            do_stitch (:obj:`bool`): enable splicing/stitching fixed parameter.
            kwargs: other options to pass through to underlying minimizer

        Returns:
            Fitted parameters or tuple of results:

                - parameters (:obj:`tensor`): fitted parameters
                - minimum (:obj:`float`): if ``return_fitted_val`` flagged, return minimized objective value
                - result (:class:`scipy.optimize.OptimizeResult`): if ``return_result_obj`` flagged
        """
        # literally just for the minimizer name to check if we're using minuit
        # so we can check if valid for uncertainty calc later
        using_minuit = hasattr(self, "name") and self.name == "minuit"

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

        # handle non-pyhf ModelConfigs
        try:
            par_names = pdf.config.par_names
        except AttributeError:
            par_names = None

        # need to remove parameters that are fixed in the fit
        if par_names and do_stitch and fixed_vals:
            for index, _ in fixed_vals:
                par_names[index] = None
            par_names = [name for name in par_names if name]

        result = self._internal_minimize(
            **minimizer_kwargs, options=kwargs, par_names=par_names
        )

        # compute uncertainties with automatic differentiation
        if not using_minuit and tensorlib.name in ['tensorflow', 'jax', 'pytorch']:
            # stitch in missing parameters (e.g. fixed parameters)
            all_pars = stitch_pars(tensorlib.astensor(result.x))
            hess_inv = tensorlib.fisher_cov(pdf, all_pars, data)
            uncertainties = tensorlib.sqrt(tensorlib.diagonal(hess_inv))
            calc_correlations = True
        else:
            hess_inv = None
            uncertainties = None
            calc_correlations = False

        # uncerts are set to 0 in here for fixed pars
        result = self._internal_postprocess(
            result,
            stitch_pars,
            using_minuit,
            return_uncertainties=return_uncertainties,
            uncertainties=uncertainties,
            hess_inv=hess_inv,
            calc_correlations=calc_correlations,
            fixed_vals=fixed_vals,
            init_pars=init_pars,
        )

        _returns = [result.x]
        if return_correlations:
            _returns.append(result.corr)
        if return_fitted_val:
            _returns.append(result.fun)
        if return_result_obj:
            _returns.append(result)
        return tuple(_returns) if len(_returns) > 1 else _returns[0]
