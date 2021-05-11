"""
Compatibility functions for translating between ROOT and pyhf
"""

import re

__all__ = ["interpret_rootname", "paramset_to_rootnames"]


def __dir__():
    return __all__


def paramset_to_rootnames(paramset):
    """
    Generates parameter names for parameters in the set as ROOT would do.

    Args:
        paramset (:obj:`pyhf.paramsets.paramset`): The parameter set.

    Returns:
        :obj:`List[str]` or :obj:`str`: The generated parameter names
        (for the non-scalar/scalar case) respectively.

    Example:

        pyhf parameter names and then the converted names for ROOT:

        * ``"lumi"`` -> ``"Lumi"``
        * unconstrained scalar parameter ``"foo"`` -> ``"foo"``
        * constrained scalar parameter ``"foo"`` -> ``"alpha_foo"``
        * non-scalar parameters ``"foo"`` -> ``"gamma_foo_i"``

        >>> import pyhf
        >>> pyhf.set_backend("numpy")
        >>> model = pyhf.simplemodels.uncorrelated_background(
        ...     signal=[12.0, 11.0], bkg=[50.0, 52.0], bkg_uncertainty=[3.0, 7.0]
        ... )
        >>> model.config.parameters
        ['mu', 'uncorr_bkguncrt']
        >>> pyhf.compat.paramset_to_rootnames(model.config.param_set("mu"))
        'mu'
        >>> pyhf.compat.paramset_to_rootnames(model.config.param_set("uncorr_bkguncrt"))
        ['gamma_uncorr_bkguncrt_0', 'gamma_uncorr_bkguncrt_1']
    """

    if paramset.name == 'lumi':
        return 'Lumi'
    if paramset.is_scalar:
        if paramset.constrained:
            return f'alpha_{paramset.name}'
        return f'{paramset.name}'
    return [f'gamma_{paramset.name}_{index}' for index in range(paramset.n_parameters)]


def interpret_rootname(rootname):
    """
    Interprets a ROOT-generated name as best as possible.

    Possible properties of a ROOT parameter are:

    * ``"constrained"``: :obj:`bool` representing if parameter is a member of a
      constrained paramset.
    * ``"is_scalar"``: :obj:`bool` representing if parameter is a member of a
      scalar paramset.
    * ``"name"``: The name of the param set.
    * ``"element"``: The index in a non-scalar param set.

    It is possible that some of the parameter names might not be determinable
    and will then hold the string value ``"n/a"``.

    Args:
        rootname (:obj:`str`): The ROOT-generated name of the parameter.

    Returns:
        :obj:`dict`: The interpreted key-value pairs.

    Example:

        >>> import pyhf
        >>> interpreted_name = pyhf.compat.interpret_rootname("gamma_foo_0")
        >>> pyhf.compat.interpret_rootname("gamma_foo_0")
        {'constrained': 'n/a', 'is_scalar': False, 'name': 'foo', 'element': 0}
        >>> pyhf.compat.interpret_rootname("alpha_foo")
        {'constrained': True, 'is_scalar': True, 'name': 'foo', 'element': 'n/a'}
        >>> pyhf.compat.interpret_rootname("Lumi")
        {'constrained': False, 'is_scalar': True, 'name': 'lumi', 'element': 'n/a'}
    """

    interpretation = {
        'constrained': 'n/a',
        'is_scalar': 'n/a',
        'name': 'n/a',
        'element': 'n/a',
    }
    if rootname.startswith('gamma_'):
        interpretation['is_scalar'] = False
        match = re.search(r'^gamma_(.+)_(\d+)$', rootname)
        if not match:
            raise ValueError(f'confusing rootname {rootname}. Please report as a bug.')
        interpretation['name'] = match.group(1)
        interpretation['element'] = int(match.group(2))
    else:
        interpretation['is_scalar'] = True

    if rootname.startswith('alpha_'):
        interpretation['constrained'] = True
        match = re.search(r'^alpha_(.+)$', rootname)
        if not match:
            raise ValueError(f'confusing rootname {rootname}. Please report as a bug.')
        interpretation['name'] = match.group(1)

    if not (rootname.startswith('alpha_') or rootname.startswith('gamma_')):
        interpretation['constrained'] = False
        interpretation['name'] = rootname

    if rootname == 'Lumi':
        interpretation['name'] = 'lumi'

    return interpretation
