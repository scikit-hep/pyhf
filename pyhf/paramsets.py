from . import get_backend
from . import exceptions

class paramset(object):
    def __init__(self, n_parameters, inits, bounds):
        self.n_parameters = n_parameters
        self.suggested_init = inits
        self.suggested_bounds = bounds

class unconstrained(paramset):
    def __init__(self, n_parameters, inits, bounds, auxdata=None, factors=None):
        super(unconstrained,self).__init__(n_parameters, inits, bounds)

class constrained_by_normal(paramset):
    def __init__(self, n_parameters, inits, bounds, auxdata, factors=None):
        super(constrained_by_normal,self).__init__(n_parameters, inits, bounds)
        self.pdf_type = 'normal'
        self.auxdata = auxdata

    def expected_data(self, pars):
        return pars

class constrained_by_poisson(paramset):
    def __init__(self, n_parameters, inits, bounds, auxdata, factors):
        super(constrained_by_poisson,self).__init__(n_parameters, inits, bounds)
        self.pdf_type = 'poisson'
        self.auxdata = auxdata
        self.factors = factors

    def expected_data(self, pars):
        tensorlib, _ = get_backend()
        return tensorlib.product(
                tensorlib.stack([pars, tensorlib.astensor(self.factors)]
            ),
            axis=0
        )

def reduce_paramset_requirements(paramset_requirements):
    reduced_paramset_requirements = {}

    # nb: normsys and histosys have different op_codes so can't currently be shared
    param_keys = ['constraint',
                  'n_parameters',
                  'op_code',
                  'inits',
                  'bounds',
                  'auxdata',
                  'factors']

    for param_name in list(paramset_requirements.keys()):
        params = paramset_requirements[param_name]

        combined_param = {}
        for param in params:
            for k in param_keys:
                combined_param.setdefault(k, set([])).add(param[k])

        for k in param_keys:
            v = combined_param[k]
            if len(v) != 1:
                raise exceptions.InvalidNameReuse("Multiple values for '{}' ({}) were found for {}. Use unique modifier names or use qualify_names=True when constructing the pdf.".format(k, list(v), param_name))
            else:
                v = list(v)[0]
                if isinstance(v, tuple): v = list(v)
                combined_param[k] = v

        reduced_paramset_requirements[param_name] = combined_param

    return reduced_paramset_requirements
