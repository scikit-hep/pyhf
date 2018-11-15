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
        sigmas = kwargs.pop('sigmas', None)
        if sigmas:
            self.sigmas = sigmas

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


def reduce_paramsets_requirements(paramsets_requirements, paramsets_user_configs):
    reduced_paramsets_requirements = {}

    paramset_keys = [
        'paramset_type',
        'n_parameters',
        'inits',
        'bounds',
        'auxdata',
        'factors',
        'sigmas',
    ]

    """
    - process all defined paramsets
    - determine the unique set of paramsets by param-name
    - if the paramset is not unique, complain
    - if the paramset is unique, build the paramset using the set() defined on its options
      - if the value is a tuple, this came from default options so convert to a list and use it
      - if the value is a list, this came from user-define options, so use it
    """
    for paramset_name in list(paramsets_requirements.keys()):
        paramset_requirements = paramsets_requirements[paramset_name]
        paramset_user_configs = paramsets_user_configs.get(paramset_name, {})

        combined_paramset = {}
        for k in paramset_keys:
            for paramset_requirement in paramset_requirements:
                v = paramset_requirement.get(k, 'undefined')
                combined_paramset.setdefault(k, set([])).add(v)

            if len(combined_paramset[k]) != 1:
                raise exceptions.InvalidNameReuse(
                    "Multiple values for '{}' ({}) were found for {}. Use unique modifier names when constructing the pdf.".format(
                        k, list(combined_paramset[k]), paramset_name
                    )
                )
            else:
                default_v = combined_paramset[k].pop()
                # get user-defined-config if it exists or set to default config
                v = paramset_user_configs.get(k, default_v)
                # if v is a tuple, it's not user-configured, so convert to list
                if v == 'undefined':
                    continue
                if isinstance(v, tuple):
                    v = list(v)
                # this implies user-configured, so check that it has the right number of elements
                elif isinstance(v, list) and default_v and len(v) != len(default_v):
                    raise exceptions.InvalidModel(
                        'Incorrect number of values ({}) for {} were configured by you, expected {}.'.format(
                            len(v), k, len(default_v)
                        )
                    )
                elif v and default_v == 'undefined':
                    raise exceptions.InvalidModel(
                        '{} does not use the {} attribute.'.format(paramset_name, k)
                    )

                combined_paramset[k] = v

        reduced_paramsets_requirements[paramset_name] = combined_paramset

    return reduced_paramsets_requirements
