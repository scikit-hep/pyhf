import logging
log = logging.getLogger(__name__)

from . import modifier
from ..paramsets import constrained_by_normal

@modifier(name='histosys', constrained=True, shared=True, op_code = 'addition')
class histosys(object):
    def __init__(self, nom_data, modifier_data):
        self.n_parameters = 1
        self.parset = constrained_by_normal(
            n_parameters = self.n_parameters, 
            inits = [0.0],
            bounds = [[-5.,5.]],
            auxdata = [0.]
        )
        assert self.n_parameters == self.parset.n_parameters
        assert self.pdf_type == self.parset.pdf_type

