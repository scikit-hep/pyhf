import re


def parset_to_rootnames(paramset):
    """
    TODO: doc
    """
    if paramset.name == 'lumi':
        return 'Lumi'
    if paramset.is_scalar:
        if paramset.constrained:
            return f'alpha_{paramset.name}'
        return f'{paramset.name}'
    else:
        return [
            f'gamma_{paramset.name}_{index}' for index in range(paramset.n_parameters)
        ]


def interpret_rootname(rootname):
    """
    TODO: doc
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
