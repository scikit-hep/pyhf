import logging
log = logging.getLogger(__name__)

from . import modifier
from ..paramsets import unconstrained

@modifier(name='shapefactor', shared=True, op_code = 'multiplication')
class shapefactor(object):
    def __init__(self, nom_data, modifier_data):
        self.n_parameters = len(nom_data)

        self.parset = unconstrained(
            self.n_parameters,
            [1.0] * self.n_parameters,
            [[0, 10]] * self.n_parameters
        )
        assert self.n_parameters == self.parset.n_parameters

    def add_sample(self, channel, sample, modifier_def):
        pass

    def apply(self, channel, sample, pars):
        return pars
