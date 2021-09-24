"""Inference for Statistical Models."""

from pyhf.infer.calculators import AsymptoticCalculator, ToyCalculator
from pyhf.exceptions import InvalidTestStatistic
from pyhf.infer.test_statistics import q0, qmu, qmu_tilde

import logging

log = logging.getLogger(__name__)

__all__ = ["create_calculator", "get_test_stat"]


def __dir__():
    return __all__


def all_pois_floating(pdf, fixed_params):
    r"""
    Check whether all POI(s) are floating (i.e. not within the fixed set).

    Args:
        pdf (~pyhf.pdf.Model): The statistical model adhering to the schema
         ``model.json``.
        fixed_params (:obj:`list` or `tensor` of :obj:`bool`): Array of
         :obj:`bool` indicating if model parameters are fixed.

    Returns:
        :obj:`bool`: The result whether all POIs are floating.
    """

    poi_fixed = fixed_params[pdf.config.poi_index]
    return not poi_fixed


def create_calculator(calctype, *args, **kwargs):
    """
    Creates a calculator object of the specified `calctype`.

    See :py:class:`~pyhf.infer.calculators.AsymptoticCalculator` and
    :py:class:`~pyhf.infer.calculators.ToyCalculator` on additional arguments
    to be specified.

    Example:

        >>> import pyhf
        >>> import numpy.random as random
        >>> random.seed(0)
        >>> model = pyhf.simplemodels.uncorrelated_background(
        ...     signal=[12.0, 11.0], bkg=[50.0, 52.0], bkg_uncertainty=[3.0, 7.0],
        ... )
        >>> observations = [51, 48]
        >>> data = observations + model.config.auxdata
        >>> mu_test = 1.0
        >>> toy_calculator = pyhf.infer.utils.create_calculator(
        ...     "toybased", data, model, ntoys=100, test_stat="qtilde", track_progress=False
        ... )
        >>> qmu_sig, qmu_bkg = toy_calculator.distributions(mu_test)
        >>> qmu_sig.pvalue(mu_test), qmu_bkg.pvalue(mu_test)
        (array(0.14), array(0.79))

    Args:
        calctype (:obj:`str`): The calculator to create. Choose either
        'asymptotics' or 'toybased'.

    Returns:
        calculator (:obj:`object`): A calculator.
    """
    return {'asymptotics': AsymptoticCalculator, 'toybased': ToyCalculator}[calctype](
        *args, **kwargs
    )


def get_test_stat(name):
    """
    Get the test statistic function by name. The following test statistics are supported:

    - :func:`~pyhf.infer.test_statistics.q0`
    - :func:`~pyhf.infer.test_statistics.qmu`
    - :func:`~pyhf.infer.test_statistics.qmu_tilde`

    Example:

        >>> from pyhf.infer import utils, test_statistics
        >>> utils.get_test_stat("q0")
        <function q0 at 0x...>
        >>> utils.get_test_stat("q0") == test_statistics.q0
        True
        >>> utils.get_test_stat("q")
        <function qmu at 0x...>
        >>> utils.get_test_stat("q") == test_statistics.qmu
        True
        >>> utils.get_test_stat("qtilde")
        <function qmu_tilde at 0x...>
        >>> utils.get_test_stat("qtilde") == test_statistics.qmu_tilde
        True

    Args:
        name (:obj:`str`): The name of the test statistic to retrieve


    Returns:
        callable: The test statistic function
    """
    _mapping = {
        "q0": q0,
        "q": qmu,
        "qtilde": qmu_tilde,
    }
    try:
        return _mapping[name]
    except KeyError:
        raise InvalidTestStatistic
