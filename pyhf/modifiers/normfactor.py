import logging
log = logging.getLogger(__name__)

from . import modifier

@modifier(name='normfactor', shared=True)
class normfactor(object):
    def __init__(self, nom_data, modifier_data):
        self.n_parameters = 1
        self.suggested_init = [1.0]
        self.suggested_bounds = [[0, 10]]

    def add_sample(self, channel, sample, modifier_def):
        pass

    def apply(self, channel, sample, pars):
        return pars
