import logging
log = logging.getLogger(__name__)

from . import modifier
from ..paramsets import constrained_by_normal

@modifier(name='staterror', shared=True, constrained=True, op_code = 'multiplication')
class staterror(object):
    @classmethod
    def required_parset(cls, n_parameters):
        return {
            'constraint': constrained_by_normal,
            'n_parameters': n_parameters,
            'modifier': cls.__name__,
            'is_constrained': cls.is_constrained,
            'is_shared': cls.is_shared,
            'op_code': cls.op_code,
            'inits': [1.] * n_parameters,
            'bounds': [[1e-10, 10.]] * n_parameters,
            'auxdata': [1.] * n_parameters,
            'factors': [] * n_parameters
        }
