"""Inference for Statistical Models."""

from .calculators import AsymptoticCalculator, ToyCalculator


def create_calculator(calctype, *args, **kwargs):
    """
    Creates a calculator object of the specified `calctype`.

    See :py:class:`~pyhf.infer.calculators.AsymptoticCalculator` and :py:class:`~pyhf.infer.calculators.ToyCalculator` on additional arguments to be specified.

    Example:

        >>> import pyhf
        >>> import numpy.random as random
        >>> random.seed(0)
        >>> model = pyhf.simplemodels.hepdata_like(
        ...     signal_data=[12.0, 11.0], bkg_data=[50.0, 52.0], bkg_uncerts=[3.0, 7.0],
        ... )
        >>> observations = [51, 48]
        >>> data = observations + model.config.auxdata
        >>> mu_test = 1.0
        >>> toy_calculator = pyhf.infer.utils.create_calculator(
        ...     "toybased", data, model, ntoys=100, track_progress=False
        ... )
        >>> qmu_sig, qmu_bkg = toy_calculator.distributions(mu_test)
        >>> qmu_sig.pvalue(mu_test), qmu_bkg.pvalue(mu_test)
        (0.14, 0.76)

    Args:
        calctype (:obj:`str`): The calculator to create. Choose either 'asymptotics' or 'toybased'.

    Returns:
        calculator (:obj:`object`): A calculator.
    """
    return {'asymptotics': AsymptoticCalculator, 'toybased': ToyCalculator,}[
        calctype
    ](*args, **kwargs)
