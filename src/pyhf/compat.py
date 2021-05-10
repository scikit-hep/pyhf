"""
Compatibility functions for translating between ROOT and pyhf
"""

import re

__all__ = [
    "paramset_to_rootnames",
    "interpret_rootname",
]


def __dir__():
    return __all__


def paramset_to_rootnames(paramset):
    """
    Generates parameter names for parameters in the set as ROOT would do.

        lumi -> Lumi
        unconstrained scalar parameter foo -> foo
        constrained scalar parameter foo -> alpha_foo
        non-scalar parameters -> gamma_foo_i

        Args:
            paramset (:obj:`pyhf.paramsets.paramset`): the parameter set

        Returns:
            rootnames (:obj:`List[str]` or `str`): the generated parameter names
                                          (for the non-scalar/scalar case)
                                          respectively
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
    Interprets a ROOT-generated name as best as possible
    Possible properties of a ROOT parameter are

    * "constrained": whether it's a member of a constrained paramset
    * "is_scalar": whether it's a member of a scalar paramset
    * "name": name of the param set
    * "element": index in a non-scalar param set

    it is possible that some of them might not be determinable
    and will then hold the strnigvalue "n/a"



        Args:
            rootname (:obj:`tensor`):

        Returns:
            interpretation (:obj:`dict`): interpreteted key-value pairs
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
            raise ValueError('confusing rootname, please report bug')
        interpretation['name'] = match.group(1)
        interpretation['element'] = int(match.group(2))
    else:
        interpretation['is_scalar'] = True

    if rootname.startswith('alpha_'):
        interpretation['constrained'] = True
        match = re.search(r'^alpha_(.+)$', rootname)
        if not match:
            raise ValueError('confusing rootname, please report bug')
        interpretation['name'] = match.group(1)

    if not (rootname.startswith('alpha_') or rootname.startswith('gamma_')):
        interpretation['constrained'] = False
        interpretation['name'] = rootname

    if rootname == 'Lumi':
        interpretation['name'] = 'lumi'

    return interpretation
