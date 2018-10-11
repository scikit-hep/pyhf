from . import get_backend
class param_set(object):
    def __init__(self, n_parameters, inits, bounds):
        self.n_parameters = n_parameters
        self.suggested_init = inits
        self.suggested_bounds = bounds

class unconstrained_set(param_set):
    pass

class normal_constrained_set(param_set):
    def __init__(self, n_parameters, inits, bounds, auxdata):
        super(normal_constrained_set,self).__init__(n_parameters, inits, bounds)
        self.pdf_type = 'normal'
        self.auxdata = auxdata

    def alphas(self, pars):
        '''the nuisance parameters correspond directly to the alpha'''
        return pars  

    def expected_data(self, pars):
        return self.alphas(pars)

class poisson_constrained_set(param_set):
    def __init__(self, n_parameters, inits, bounds, auxdata, factors):
        super(poisson_constrained_set,self).__init__(n_parameters, inits, bounds)
        self.pdf_type = 'poisson'
        self.auxdata = auxdata
        self.factors = factors

    def alphas(self, pars):
        tensorlib, _ = get_backend()
        return tensorlib.product(
                tensorlib.stack([pars, tensorlib.astensor(self.factors)]
            ),
            axis=0
        )

    def expected_data(self, pars):
        return self.alphas(pars)

