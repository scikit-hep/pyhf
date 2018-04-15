import logging
log = logging.getLogger(__name__)

from six import with_metaclass
from . import IModifier
from .. import tensorlib

class normfactor(with_metaclass(IModifier, object)):
    is_constraint = False

    def __init__(self):
        self.n_parameters = 1
        self.suggested_init = [1.0]
        self.suggested_bounds = [[0, 10]]

    def alphas(self, pars):
        raise NotImplementedError

    def expected_data(self, pars):
        raise NotImplementedError

    def pdf(self, a, alpha):
        raise NotImplementedError
