import logging
log = logging.getLogger(__name__)

from . import modifier
from ..paramsets import constrained_by_poisson

@modifier(name='shapesys', constrained=True, pdf_type='poisson', op_code = 'multiplication')
class shapesys(object):
    @classmethod
    def required_parset(cls, nom_data):
        n_parameters = len(nom_data)
        return {
            'parset': constrained_by_poisson,
            'n_parameters': n_parameters,
            'inits': [1.0] * n_parameters,
            'bounds' :[[1e-10, 10.]] * n_parameters,
            'auxdata': [-1.] * n_parameters,
            'factors': [-1.] * n_parameters,
            'modifier': cls.__name__,
            'is_constrained': cls.is_constrained,
            'is_shared': cls.is_shared,
            'op_code': cls.op_code
        }
