import logging
log = logging.getLogger(__name__)

from . import modifier
from ..paramsets import constrained_by_normal

@modifier(name='staterror', shared=True, constrained=True, op_code = 'multiplication')
class staterror(object):
    def __init__(self, nom_data, modifier_data):
        self.n_parameters = len(nom_data)

        self.parset = constrained_by_normal(
            n_parameters = self.n_parameters,
            inits = [1.] * self.n_parameters,
            bounds = [[1e-10, 10.]] * self.n_parameters,
            auxdata = [1.] * self.n_parameters
        )
        assert self.n_parameters == self.parset.n_parameters
        assert self.pdf_type == self.parset.pdf_type

