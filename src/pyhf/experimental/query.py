""" Supporting Query Language functionality for PDFs """
from .. import utils

import fnmatch


def parameter_indices(model, expression):
    __pattern = '__PYHFPARPATTERN__'
    expression = expression.replace('*', __pattern)
    par_name, subscript = utils.parse_parameter_name(expression)

    par_name = par_name.replace(__pattern, '*')
    indices = []
    for parameter in fnmatch.filter(model.config.parameters, par_name):
        par_slice = model.config.par_slice(parameter)
        par_indices = list(range(par_slice.stop)[par_slice])[subscript]
        if isinstance(par_indices, list):
            indices = indices + par_indices
        else:
            indices.append(par_indices)

    if not isinstance(subscript, slice) and len(indices) == 1:
        return indices[0]

    return indices
