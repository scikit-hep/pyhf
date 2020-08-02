from .. import default_backend


class paramset(object):
    def __init__(self, **kwargs):
        self.n_parameters = kwargs.pop('n_parameters')
        self.suggested_init = kwargs.pop('inits')
        self.suggested_bounds = kwargs.pop('bounds')
        self.fixed = kwargs.pop('fixed')


class unconstrained(paramset):
    def __init__(self, **kwargs):
        super(unconstrained, self).__init__(**kwargs)
        self.constrained = False


class constrained_paramset(paramset):
    def __init__(self, **kwargs):
        super(constrained_paramset, self).__init__(**kwargs)
        self.constrained = True


class constrained_by_normal(constrained_paramset):
    def __init__(self, **kwargs):
        super(constrained_by_normal, self).__init__(**kwargs)
        self.pdf_type = 'normal'
        self.auxdata = kwargs.pop('auxdata')
        sigmas = kwargs.pop('sigmas', None)
        if sigmas:
            self.sigmas = sigmas

    def width(self):
        try:
            return self.sigmas
        except AttributeError:
            return [1.0] * self.n_parameters


class constrained_by_poisson(constrained_paramset):
    def __init__(self, **kwargs):
        super(constrained_by_poisson, self).__init__(**kwargs)
        self.pdf_type = 'poisson'
        self.auxdata = kwargs.pop('auxdata')
        factors = kwargs.pop('factors')
        if factors:
            self.factors = factors

    def width(self):
        try:
            return default_backend.sqrt(
                1.0 / default_backend.astensor(self.factors)
            ).tolist()
        except AttributeError:
            raise RuntimeError('need to know rate factor to compu')
