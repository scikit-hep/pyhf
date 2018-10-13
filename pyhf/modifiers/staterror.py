import logging
log = logging.getLogger(__name__)

from . import modifier
from ..paramsets import constrained_by_normal

@modifier(name='staterror', shared=True, constrained=True, op_code = 'multiplication')
class staterror(object):
    @classmethod
    def create_parset(cls, nom_data):
        n_parameters = len(nom_data)
        parset = constrained_by_normal(
            n_parameters = n_parameters,
            inits = [1.] * n_parameters,
            bounds = [[1e-10, 10.]] * n_parameters,
            auxdata = [1.] * n_parameters
        )
        assert n_parameters == parset.n_parameters
        assert cls.pdf_type == parset.pdf_type
        return parset
