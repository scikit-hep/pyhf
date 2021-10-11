from pyhf import default_backend

__all__ = [
    "constrained_by_normal",
    "constrained_by_poisson",
    "constrained_paramset",
    "paramset",
    "unconstrained",
]


def __dir__():
    return __all__


class paramset:
    def __init__(self, **kwargs):
        self.name = kwargs.pop('name')
        self.n_parameters = kwargs.pop('n_parameters')
        self.suggested_init = kwargs.pop('inits')
        self.suggested_bounds = kwargs.pop('bounds')
        self.suggested_fixed = kwargs.pop('fixed')
        self.is_scalar = kwargs.pop('is_scalar')
        if self.is_scalar and not (self.n_parameters == 1):
            raise ValueError(
                f'misconfigured parameter set {self.name}. Scalar but N>1 parameters.'
            )


class unconstrained(paramset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.constrained = False


class constrained_paramset(paramset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.constrained = True


class constrained_by_normal(constrained_paramset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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
        super().__init__(**kwargs)
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
