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


def reduce_paramset_requirements(paramset_requirements, paramsets_user_configs):
    reduced_paramset_requirements = {}

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
        param_user_configs = paramsets_user_configs.get(param_name, {})

        combined_param = {}
        for param in params:
            for k in param_keys:
                v = param.get(k)
                combined_param.setdefault(k, set([])).add(v)

        for k in param_keys:
            if len(combined_param[k]) != 1 and k != 'op_code':
                raise exceptions.InvalidNameReuse(
                    "Multiple values for '{}' ({}) were found for {}. Use unique modifier names when constructing the pdf.".format(
                        k, list(combined_param[k]), param_name
                    )
                )
            else:
                default_v = combined_param[k].pop()
                # get user-defined-config if it exists or set to default config
                v = param_user_configs.get(k, default_v)
                # if v is a tuple, it's not user-configured, so convert to list
                if isinstance(v, tuple):
                    v = list(v)
                # this implies user-configured, so check that it has the right number of elements
                elif isinstance(v, list) and len(v) != len(default_v):
                    raise exceptions.InvalidModel(
                        'Incorrect number of values ({}) for {} were configured by you, expected {}.'.format(
                            len(v), k, len(default_v)
                        )
                    )

                combined_param[k] = v

        reduced_paramset_requirements[param_name] = combined_param

    return reduced_paramset_requirements
