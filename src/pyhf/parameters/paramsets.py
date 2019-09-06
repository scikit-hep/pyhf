from .. import get_backend


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
        sh = tensorlib.shape(pars)
        # if batched, tile the factors as they only depend on background uncertainty
        if len(sh) > 1:
            fc = tensorlib.tile(
                tensorlib.reshape(tensorlib.astensor(self.factors), (1, -1)), (sh[0], 1)
            )
            return pars * fc
        return tensorlib.product(
            tensorlib.stack([pars, tensorlib.astensor(self.factors)]), axis=0
        )
