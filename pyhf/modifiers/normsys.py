import logging
log = logging.getLogger(__name__)

from . import modifier
from ..paramsets import constrained_by_normal

@modifier(name='normsys', constrained=True, shared=True, op_code = 'multiplication')
class normsys(object):
    @classmethod
    def required_parset(cls, nom_data):
        n_parameters     = 1
        return {
            'parset': constrained_by_normal,
            'n_parameters': n_parameters,
            'inits': [0.0] * n_parameters,
            'bounds': [[-5., 5.]] * n_parameters,
            'auxdata': [0.],
            'factors': None,
            'modifier': cls.__name__,
            'is_constrained': cls.is_constrained,
            'is_shared': cls.is_shared,
            'op_code': cls.op_code
        }
