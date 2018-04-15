import logging
log = logging.getLogger(__name__)

from six import with_metaclass
from . import IModifier
from .. import tensorlib

class shapefactor(with_metaclass(IModifier, object)):
    is_constraint = False

    def __init__(self, nom_data, modifier_data):
        self.n_parameters = len(nom_data)
        self.suggested_init = [1.0] * self.n_parameters
        self.suggested_bounds = [[0, 10]] * self.n_parameters

    def alphas(self, pars):
        raise NotImplementedError

    def expected_data(self, pars):
        raise NotImplementedError

    def pdf(self, a, alpha):
        raise NotImplementedError
