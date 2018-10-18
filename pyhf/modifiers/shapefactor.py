import logging
log = logging.getLogger(__name__)

from . import modifier
from ..paramsets import unconstrained

@modifier(name='shapefactor', shared=True, op_code = 'multiplication')
class shapefactor(object):
    @classmethod
    def required_parset(cls, n_parameters):
        return {
            'parset': unconstrained,
            'n_parameters': n_parameters,
            'inits': [1.0] * n_parameters,
            'bounds' :[[0, 10]] * n_parameters,
            'auxdata': None,
            'factors': None,
            'modifier': cls.__name__,
            'is_constrained': cls.is_constrained,
            'is_shared': cls.is_shared,
            'op_code': cls.op_code
        }
