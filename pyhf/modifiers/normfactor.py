from six import with_metaclass
from . import IModifier
from .. import tensorlib

class normfactor(with_metaclass(IModifier, object)):
    is_constraint = False

    @staticmethod
    def suggested_init(n_parameters):
        return [1.0]

    @staticmethod
    def suggested_bounds(n_parameters):
        return [[0, 10]]

    def __init__(self):
        raise NotImplementedError

    def add_sample(self, channel, sample, modifier_data):
        raise NotImplementedError

    def alphas(self, pars):
        raise NotImplementedError

    def expected_data(self, pars):
        raise NotImplementedError

    def pdf(self, a, alpha):
        raise NotImplementedError
