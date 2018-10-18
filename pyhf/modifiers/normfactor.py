import logging
log = logging.getLogger(__name__)

from . import modifier
from ..paramsets import unconstrained

@modifier(name='normfactor', shared=True, op_code = 'multiplication')
class normfactor(object):
    @classmethod
    def required_parset(cls, nom_data):
        n_parameters = 1
        return {
            'parset': unconstrained,
            'n_parameters': n_parameters,
            'inits': [1.0] * n_parameters,
            'bounds': [[0, 10]] * n_parameters,
            'auxdata': None,
            'factors': None,
            'modifier': cls.__name__,
            'is_constrained': cls.is_constrained,
            'is_shared': cls.is_shared,
            'op_code': cls.op_code
        }
