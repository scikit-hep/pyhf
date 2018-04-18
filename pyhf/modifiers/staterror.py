import logging
log = logging.getLogger(__name__)

from . import modifier
from .. import get_backend

@modifier(name='statsys', constrained=True)
class staterror(object):
    def __init__(self, nom_data, modifier_data):
        self.n_parameters = len(nom_data)
        self.suggested_init = [1.0] * self.n_parameters
        self.suggested_bounds = [[0, 10]] * self.n_parameters

    def alphas(self, pars):
        raise NotImplementedError

    def pdf(self, a, alpha):
        raise NotImplementedError

    def expected_data(self, pars):
        raise NotImplementedError

    def add_sample(self, channel, sample, modifier_def):
        log.info('adding sample to staterror')
