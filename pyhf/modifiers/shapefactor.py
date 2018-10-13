import logging
log = logging.getLogger(__name__)

from . import modifier
from ..paramsets import unconstrained

@modifier(name='shapefactor', shared=True, op_code = 'multiplication')
class shapefactor(object):
    @classmethod
    def create_parset(cls, nom_data):
        n_parameters = len(nom_data)
        parset = unconstrained(
            n_parameters,
            [1.0] * n_parameters,
            [[0, 10]] * n_parameters
        )
        assert n_parameters == parset.n_parameters
        return parset
