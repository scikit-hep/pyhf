from . import get_backend
from . import exceptions


class paramset(object):
    def __init__(self, **kwargs):
        self.n_parameters = kwargs.pop('n_parameters')
        self.suggested_init = kwargs.pop('inits')
        self.suggested_bounds = kwargs.pop('bounds')


class unconstrained(paramset):
    pass


class constrained_by_normal(paramset):
    def __init__(self, **kwargs):
        super(constrained_by_normal, self).__init__(**kwargs)
        self.pdf_type = 'normal'
        self.auxdata = kwargs.pop('auxdata')

    def expected_data(self, pars):
        return pars


class constrained_by_poisson(paramset):
    def __init__(self, **kwargs):
        super(constrained_by_poisson, self).__init__(**kwargs)
        self.pdf_type = 'poisson'
        self.auxdata = kwargs.pop('auxdata')
        self.factors = kwargs.pop('factors')

    def expected_data(self, pars):
        tensorlib, _ = get_backend()
        return tensorlib.product(
            tensorlib.stack([pars, tensorlib.astensor(self.factors)]), axis=0
        )


def reduce_paramset_requirements(paramset_requirements):
    reduced_paramset_requirements = {}

    # nb: normsys and histosys have different op_codes so can't currently be shared
    param_keys = [
        'paramset_type',
        'n_parameters',
        'op_code',
        'inits',
        'bounds',
        'auxdata',
        'factors',
    ]

    for param_name in list(paramset_requirements.keys()):
        params = paramset_requirements[param_name]

        combined_param = {}
        for param in params:
            for k in param_keys:
                v = param.get(k)
                # need to convert lists to tuples
                if k in ['inits', 'auxdata', 'factors']:
                    v = tuple(v)
                # need to convert lists of lists to tuples
                elif k in ['bounds']:
                    v = tuple(map(tuple, v))
                combined_param.setdefault(k, set([])).add(v)

        for k in param_keys:
            if len(combined_param[k]) != 1 and k != 'op_code':
                raise exceptions.InvalidNameReuse(
                    "Multiple values for '{}' ({}) were found for {}. Use unique modifier names when constructing the pdf.".format(
                        k, list(combined_param[k]), param_name
                    )
                )
            else:
                v = combined_param[k].pop()
                if isinstance(v, tuple):
                    v = list(v)
                combined_param[k] = v

        reduced_paramset_requirements[param_name] = combined_param

    return reduced_paramset_requirements
