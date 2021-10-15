"""Helper Classes for use of automatic differentiation."""
from pyhf.tensor.manager import get_backend
from pyhf import exceptions
from pyhf.optimize.common import shim

import logging

log = logging.getLogger(__name__)


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
                f"Unsupported kwargs were passed in: {list(kwargs.keys())}."
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

    def _internal_postprocess(self, fitresult, stitch_pars, return_uncertainties=False):
        """
        Post-process the fit result.

        Returns:
            fitresult (scipy.optimize.OptimizeResult): A modified version of the fit result.
        """
        tensorlib, _ = get_backend()

        # stitch in missing parameters (e.g. fixed parameters)
        fitted_pars = stitch_pars(tensorlib.astensor(fitresult.x))

        # check if uncertainties were provided (and stitch just in case)
        uncertainties = getattr(fitresult, 'unc', None)
        if uncertainties is not None:
            # extract number of fixed parameters
            num_fixed_pars = len(fitted_pars) - len(fitresult.x)
            # stitch in zero-uncertainty for fixed values
            uncertainties = stitch_pars(
                tensorlib.astensor(uncertainties),
                stitch_with=tensorlib.zeros(num_fixed_pars),
            )
            if return_uncertainties:
                fitted_pars = tensorlib.stack([fitted_pars, uncertainties], axis=1)

        correlations = getattr(fitresult, 'corr', None)
        if correlations is not None:
            _zeros = tensorlib.zeros(num_fixed_pars)
            # possibly a more elegant way to do this
            stitched_columns = [
                stitch_pars(tensorlib.astensor(column), stitch_with=_zeros)
                for column in zip(*correlations)
            ]
            stitched_rows = [
                stitch_pars(tensorlib.astensor(row), stitch_with=_zeros)
                for row in zip(*stitched_columns)
            ]
            correlations = tensorlib.stack(stitched_rows, axis=1)

        fitresult.x = fitted_pars
        fitresult.fun = tensorlib.astensor(fitresult.fun)
        fitresult.unc = uncertainties
        fitresult.corr = correlations

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
            par_names = pdf.config.par_names()
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
        result = self._internal_postprocess(
            result, stitch_pars, return_uncertainties=return_uncertainties
        )

        _returns = [result.x]
        if return_correlations:
            _returns.append(result.corr)
        if return_fitted_val:
            _returns.append(result.fun)
        if return_result_obj:
            _returns.append(result)
        return tuple(_returns) if len(_returns) > 1 else _returns[0]
