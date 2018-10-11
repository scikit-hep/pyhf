from . import get_backend
class paramset(object):
    def __init__(self, n_parameters, inits, bounds):
        self.n_parameters = n_parameters
        self.suggested_init = inits
        self.suggested_bounds = bounds

class unconstrained(paramset):
    pass

class constrained_by_normal(paramset):
    def __init__(self, n_parameters, inits, bounds, auxdata):
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

