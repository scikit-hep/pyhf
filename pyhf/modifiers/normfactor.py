import logging
log = logging.getLogger(__name__)

from six import with_metaclass
from . import modifier
from .. import get_backend

@modifier
class normfactor(object):
    is_constrained = False

    def __init__(self, nom_data, modifier_data):
        self.n_parameters = 1
        self.suggested_init = [1.0]
        self.suggested_bounds = [[0, 10]]

    def alphas(self, pars):
        raise NotImplementedError

    def expected_data(self, pars):
        raise NotImplementedError

    def pdf(self, a, alpha):
        raise NotImplementedError

    def add_sample(self, channel, sample, modifier_data):
        pass
