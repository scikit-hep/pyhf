import logging
log = logging.getLogger(__name__)

from . import modifier
from ..paramsets import constrained_by_poisson

@modifier(name='shapesys', constrained=True, pdf_type='poisson', op_code = 'multiplication')
class shapesys(object):
    @classmethod
    def create_parset(cls, nom_data):
        n_parameters = len(nom_data)
        parset = constrained_by_poisson(
            n_parameters = n_parameters,
            inits = [1.0] * n_parameters,
            bounds = [[1e-10, 10.]] * n_parameters,
            auxdata = [-1.]*n_parameters, 
            factors = [-1.]*n_parameters
        )#auxdata and factors *must* be set be the combiend modifier at some point

        assert n_parameters == parset.n_parameters
        assert cls.pdf_type == parset.pdf_type
        return parset

