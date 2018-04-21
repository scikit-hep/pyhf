import logging
log = logging.getLogger(__name__)

from . import modifier

@modifier(name='shapefactor', shared=True)
class shapefactor(object):
    def __init__(self, nom_data, modifier_data):
        self.n_parameters = len(nom_data)
        self.suggested_init = [1.0] * self.n_parameters
        self.suggested_bounds = [[0, 10]] * self.n_parameters

    def add_sample(self, channel, sample, modifier_def):
        pass

    def apply(self, channel, sample, pars):
        return pars
