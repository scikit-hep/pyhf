import logging
log = logging.getLogger(__name__)

from . import modifier
from ..constraints import param_set


@modifier(name='normfactor', shared=True, op_code = 'multiplication')
class normfactor(object):
    def __init__(self, nom_data, modifier_data):
        self.n_parameters = 1
        self.parset = param_set(
            self.n_parameters,
            [1.0],
            [[0, 10]]
        )
        assert self.n_parameters == self.parset.n_parameters
    
    def add_sample(self, channel, sample, modifier_def):
        pass

    def apply(self, channel, sample, pars):
        return pars
