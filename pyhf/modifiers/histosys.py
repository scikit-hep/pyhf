import logging
log = logging.getLogger(__name__)

from . import modifier
from ..paramsets import constrained_by_normal

@modifier(name='histosys', constrained=True, shared=True, op_code = 'addition')
class histosys(object):
    @classmethod
    def create_parset(cls, nom_data):
        n_parameters = 1
        parset = constrained_by_normal(
            n_parameters = n_parameters, 
            inits = [0.0],
            bounds = [[-5.,5.]],
            auxdata = [0.]
        )
        assert n_parameters == parset.n_parameters
        assert cls.pdf_type == parset.pdf_type
        return parset
