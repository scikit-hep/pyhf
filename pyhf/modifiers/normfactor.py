import logging
log = logging.getLogger(__name__)

from . import modifier
from ..paramsets import unconstrained

@modifier(name='normfactor', shared=True, op_code = 'multiplication')
class normfactor(object):
    def __init__(self, nom_data, modifier_data):
        self.n_parameters = 1

        self.parset = unconstrained(
            self.n_parameters,
            [1.0] * self.n_parameters,
            [[0, 10]] * self.n_parameters
        )
        assert self.n_parameters == self.parset.n_parameters
    