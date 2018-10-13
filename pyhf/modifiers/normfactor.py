import logging
log = logging.getLogger(__name__)

from . import modifier
from ..paramsets import unconstrained

@modifier(name='normfactor', shared=True, op_code = 'multiplication')
class normfactor(object):
    @classmethod
    def create_parset(cls, nom_data):
        n_parameters = 1
        parset = unconstrained(
            n_parameters,
            [1.0] * n_parameters,
            [[0, 10]] * n_parameters
        )
        assert n_parameters == parset.n_parameters
        return parset
    