"""Helper Classes for use of automatic differentiation."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import scipy.optimize

from pyhf import exceptions
from pyhf.optimize.common import shim
from pyhf.tensor.manager import get_backend

log = logging.getLogger(__name__)


def _minuit_at_limit_flags(
    minuit: Any, npars: int, fixed_idx: Sequence[int]
) -> list[bool]:
    """
    Apply iminuit's at-limit criterion to each parameter: a free bounded
    parameter is at a limit when it is within half its uncertainty of a bound.

    iminuit computes this internally but does not currently expose it per
    parameter --- only the OR-aggregated result is public, as
    :attr:`iminuit.util.FMin.has_parameters_at_limit` (as of iminuit v2.32.0,
    cf. ``iminuit/util.py`` in :class:`iminuit.util.FMin`) --- so the same
    criterion is re-evaluated here from the public per-parameter data in
    ``minuit.params``. If iminuit gains a per-parameter API for this (e.g.
    ``Param.at_limit``), this function can be replaced by it.

    Returns:
        :obj:`list` of :obj:`bool`: flags aligned with the full set of model parameters
    """
    minimizer_flags = []
    for param in minuit.params:
        if param.is_fixed or not param.has_limits:
            minimizer_flags.append(False)
            continue
        lower = param.lower_limit if param.has_lower_limit else -np.inf
        upper = param.upper_limit if param.has_upper_limit else np.inf
        minimizer_flags.append(
            min(param.value - lower, upper - param.value) < 0.5 * param.error
        )
    if len(minimizer_flags) == npars:
        return minimizer_flags
    # do_stitch=True: minuit saw only the free parameters, so map the flags
    # back to model parameter indices
    fixed = set(fixed_idx)
    variable_idx = [index for index in range(npars) if index not in fixed]
    flags = [False] * npars
    for model_index, minimizer_flag in zip(variable_idx, minimizer_flags):
        flags[model_index] = minimizer_flag
    return flags


def _at_bound_warning_messages(
    fitted_pars: Sequence[float],
    par_bounds: Sequence[Sequence[float | None]],
    fixed_idx: Sequence[int],
    par_names: Sequence[str] | None,
    at_limit: Sequence[bool] | None = None,
    rtol: float = 1e-8,
) -> list[str]:
    """
    Build warning messages for free fitted parameters that are at a bound.

    Args:
        fitted_pars (:obj:`list` of :obj:`float`): the full set of fitted model parameters
        par_bounds (:obj:`list` of :obj:`list`/:obj:`tuple`): bounds for the full set of
            model parameters. A bound that is ``None`` or non-finite (e.g. ``inf``) is
            treated as unbounded on that side.
        fixed_idx (:obj:`sequence` of :obj:`int`): indices of parameters held constant in
            the fit, which are skipped (a parameter fixed at a bound is deliberate)
        par_names (:obj:`list` of :obj:`str` or :obj:`None`): names of the full set of
            model parameters, used to identify parameters. If ``None``, parameters are
            identified by index only.
        at_limit (:obj:`list` of :obj:`bool` or :obj:`None`): optional per-parameter
            at-limit flags from the optimizer's own determination (see
            :func:`_minuit_at_limit_flags`). Parameters flagged here but not caught
            by the tolerance check are reported with the optimizer's criterion
            spelled out, as they need not be numerically at a bound.
        rtol (:obj:`float`): relative tolerance for deciding that a fitted parameter is
            at a bound. The tolerance scales with the bound range (or with
            ``max(1, |bound|)`` if only one side is bounded), so that it stays meaningful
            for both wide bounds and bounds whose endpoints are small --- pyhf's own
            ``shapesys`` and ``staterror`` gammas default to ``(1e-10, 10)``. Optimizers
            stop near but not exactly at bounds (scipy's SLSQP rails within O(1e-14)),
            so exact comparison would miss them. Default is ``1e-8``.

    Returns:
        :obj:`list` of :obj:`str`: one message per parameter at a bound
    """

    def _par_label(par_index: int) -> str:
        return (
            f"'{par_names[par_index]}' (index {par_index})"
            if par_names is not None and par_index < len(par_names)
            else f"at index {par_index}"
        )

    messages: list[str] = []
    fixed = set(fixed_idx)
    for par_index, (fitted_par, bounds) in enumerate(zip(fitted_pars, par_bounds)):
        if par_index in fixed:
            continue
        # a None or non-finite bound is unbounded on that side
        raw_lower, raw_upper = bounds
        lower = (
            raw_lower if raw_lower is not None and math.isfinite(raw_lower) else None
        )
        upper = (
            raw_upper if raw_upper is not None and math.isfinite(raw_upper) else None
        )
        # display the bounds as provided, not as normalized
        lower_str = "None" if raw_lower is None else f"{raw_lower:g}"
        upper_str = "None" if raw_upper is None else f"{raw_upper:g}"
        bounds_str = f"bounds=({lower_str}, {upper_str})"

        if not math.isfinite(fitted_par):
            messages.append(
                f"fit result for parameter {_par_label(par_index)} is not finite: value={fitted_par!r}, {bounds_str}"
            )
            continue
        # scale the tolerance with the bound range so that it stays meaningful
        # for both wide bounds and bounds with small endpoints
        if lower is not None and upper is not None:
            tolerance = rtol * abs(upper - lower)
        else:
            one_sided = lower if lower is not None else upper
            tolerance = (
                rtol * max(1.0, abs(one_sided)) if one_sided is not None else None
            )
        if (
            lower is not None
            and lower == upper
            and abs(fitted_par - lower) <= tolerance
        ):
            # bounds that pin the parameter to a single value are deliberate,
            # like the fixed parameters skipped above, but only exonerate
            # the parameter if it actually sits at the pinned value
            continue
        at_lower = lower is not None and fitted_par <= lower + tolerance
        at_upper = upper is not None and fitted_par >= upper - tolerance
        optimizer_at_limit = (
            at_limit is not None
            and par_index < len(at_limit)
            and bool(at_limit[par_index])
        )
        if not (at_lower or at_upper or optimizer_at_limit):
            continue

        if (lower is not None and fitted_par < lower) or (
            upper is not None and fitted_par > upper
        ):
            # Being strictly outside the bounds is worse than being at one.
            # The minimization returned a point the bounds should have excluded.
            messages.append(
                f"fit result for parameter {_par_label(par_index)} is outside its bounds: value={fitted_par!r}, {bounds_str}"
            )
        elif at_lower or at_upper:
            messages.append(
                f"fit result for parameter {_par_label(par_index)} is at a bound: value={fitted_par!r}, {bounds_str}"
            )
        else:
            # flagged only by the optimizer's own statistical criterion, so
            # the value need not be numerically at a bound
            messages.append(
                f"fit result for parameter {_par_label(par_index)} is within half its uncertainty of a bound (iminuit at-limit criterion): value={fitted_par!r}, {bounds_str}"
            )
    return messages


class OptimizerMixin:
    """Mixin Class to build optimizers."""

    __slots__ = ["maxiter", "verbose"]

    def __init__(self, **kwargs):
        """
        Create an optimizer.

        Args:
            maxiter (:obj:`int`): maximum number of iterations. Default is 100000.
            verbose (:obj:`int`): verbose output level during minimization. Default is off (0).
        """
        self.maxiter = kwargs.pop("maxiter", 100000)
        self.verbose = kwargs.pop("verbose", 0)

        if kwargs:
            msg = f"Unsupported kwargs were passed in: {list(kwargs)}."
            raise exceptions.Unsupported(msg)

    def _internal_minimize(
        self,
        func,
        x0,
        do_grad=False,
        bounds=None,
        fixed_vals=None,
        options=None,
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
        if options is None:
            options = {}
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
        except AssertionError as exc:
            log.exception(result)
            raise exceptions.FailedMinimization(result) from exc
        return result

    def _internal_postprocess(
        self,
        fitresult,
        stitch_pars,
        *,
        par_bounds=None,
        fixed_idx=(),
        par_names=None,
        return_uncertainties=False,
    ):
        """
        Post-process the fit result.

        Args:
            fitresult (scipy.optimize.OptimizeResult): Fit result from :func:`_internal_minimize`
            stitch_pars (:obj:`func`): callable that stitches fixed parameters into the unfixed parameters
            par_bounds (:obj:`list` of :obj:`list`/:obj:`tuple`): The extrema of values the full set of
                model parameters are allowed to reach in the fit.
                The shape should be ``(n, 2)`` for ``n`` model parameters.
                If ``None`` (the default), the check warning about fitted parameters at their
                bounds is skipped.
            fixed_idx (:obj:`sequence` of :obj:`int`): The indices of the model parameters held
                constant in the fit, which are excluded from the at-bound check.
            par_names (:obj:`list` of :obj:`str`): The names of the full set of model parameters,
                used to identify parameters in the at-bound warning. If ``None``, parameters are
                identified by index only.
            return_uncertainties (:obj:`bool`): Return uncertainties on the fitted parameters. Default is off (``False``).

        Returns:
            fitresult (scipy.optimize.OptimizeResult): A modified version of the fit result.
        """
        tensorlib, _ = get_backend()

        # stitch in missing parameters (e.g. fixed parameters)
        fitted_pars = stitch_pars(tensorlib.astensor(fitresult.x))

        if par_bounds is not None and log.isEnabledFor(logging.WARNING):
            # fitted_pars covers the full set of model parameters, so it aligns
            # with par_bounds whether or not fixed parameters were stripped from
            # fitresult.x for the minimization (do_stitch)
            fitted_pars_list = tensorlib.tolist(fitted_pars)
            npars = len(fitted_pars_list)
            par_bounds = list(par_bounds)
            if len(par_bounds) != npars:
                log.warning(
                    "length of par_bounds (%d) does not match the number of model parameters (%d), skipping check for parameters at bounds",
                    len(par_bounds),
                    npars,
                )
            else:
                minuit = getattr(fitresult, "minuit", None)
                # the O(1) aggregate gates the per-parameter walk
                at_limit = (
                    _minuit_at_limit_flags(minuit, npars, fixed_idx)
                    if minuit is not None and minuit.fmin.has_parameters_at_limit
                    else None
                )
                at_bound_messages = _at_bound_warning_messages(
                    fitted_pars_list,
                    par_bounds,
                    fixed_idx,
                    par_names,
                    at_limit=at_limit,
                )
                if at_bound_messages:
                    # a single warning per fit to avoid excessively logging in
                    # pseudoexperiment studies
                    log.warning("\n".join(at_bound_messages))

        # check if uncertainties were provided (and stitch just in case)
        uncertainties = getattr(fitresult, "unc", None)
        if uncertainties is not None:
            # extract number of fixed parameters
            num_fixed_pars = len(fitted_pars) - len(fitresult.x)

            # FIXME: Set uncertainties for fixed parameters to 0 manually
            # https://github.com/scikit-hep/iminuit/issues/762
            # https://github.com/scikit-hep/pyhf/issues/1918
            # https://github.com/scikit-hep/cabinetry/pull/346
            uncertainties = np.where(fitresult.minuit.fixed, 0.0, uncertainties)

            # stitch in zero-uncertainty for fixed values
            uncertainties = stitch_pars(
                tensorlib.astensor(uncertainties),
                stitch_with=tensorlib.zeros(num_fixed_pars),
            )
            if return_uncertainties:
                fitted_pars = tensorlib.stack([fitted_pars, uncertainties], axis=1)

        correlations = getattr(fitresult, "corr", None)
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

        When ``par_bounds`` is provided, a ``WARNING`` log message is emitted for
        any free fitted parameter that is at (or outside) a bound, as such
        parameters can indicate problems with the fit. With the minuit optimizer
        parameters that iminuit reports as being at a limit --- within half their
        uncertainty of a bound, which need not be numerically at it --- are
        reported as well, under distinct wording.

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

        if isinstance(par_bounds, scipy.optimize.Bounds):
            # scipy accepts a Bounds instance, so can use the optimizer
            # API. Normalize to pairs (lb/ub broadcast to the number of
            # parameters) so the stitching and postprocessing paths, which
            # index bounds per parameter, do not choke on it
            par_bounds = list(
                zip(
                    np.broadcast_to(par_bounds.lb, len(init_pars)),
                    np.broadcast_to(par_bounds.ub, len(init_pars)),
                )
            )

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

        # the minimizer only sees the free parameters when stitching, so
        # remove the names of parameters that are fixed in the fit (without
        # mutating par_names, which is needed in full in postprocessing)
        minimizer_par_names = par_names
        if par_names and do_stitch and fixed_vals:
            fixed = {index for index, _ in fixed_vals}
            minimizer_par_names = [
                name for index, name in enumerate(par_names) if index not in fixed
            ]

        result = self._internal_minimize(
            **minimizer_kwargs, options=kwargs, par_names=minimizer_par_names
        )
        result = self._internal_postprocess(
            result,
            stitch_pars,
            par_bounds=par_bounds,
            fixed_idx=[index for index, _ in fixed_vals or []],
            par_names=par_names,
            return_uncertainties=return_uncertainties,
        )

        _returns = [result.x]
        if return_correlations:
            _returns.append(result.corr)
        if return_fitted_val:
            _returns.append(result.fun)
        if return_result_obj:
            _returns.append(result)
        return tuple(_returns) if len(_returns) > 1 else _returns[0]
