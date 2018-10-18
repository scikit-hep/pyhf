import logging
log = logging.getLogger(__name__)

from . import modifier
from ..paramsets import constrained_by_normal

@modifier(name='histosys', constrained=True, shared=True, op_code = 'addition')
class histosys(object):
    @classmethod
    def required_parset(cls, n_parameters):
        return {
            'parset': constrained_by_normal,
            'n_parameters': 1,
            'inits': [0.0],
            'bounds': [[-5.,5.]],
            'auxdata': [0.],
            'factors': None,
            'modifier': cls.__name__,
            'is_constrained': cls.is_constrained,
            'is_shared': cls.is_shared,
            'op_code': cls.op_code
        }
