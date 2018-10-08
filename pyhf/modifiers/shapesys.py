import logging
log = logging.getLogger(__name__)

from . import modifier
from .. import get_backend
from ..constraints import factor_poisson_constraint

@modifier(name='shapesys', constrained=True, pdf_type='poisson', op_code = 'multiplication')
class shapesys(object):
    def __init__(self, nom_data, modifier_data):
        self.n_parameters = len(nom_data)

        self.bkg_over_db_squared = []
        for b, deltab in zip(nom_data, modifier_data):
            bkg_over_bsq = b * b / deltab / deltab  # tau*b
            log.info('shapesys for b,delta b (%s, %s) -> tau*b = %s',
                     b, deltab, bkg_over_bsq)
            self.bkg_over_db_squared.append(bkg_over_bsq)
        self.constraint = factor_poisson_constraint(
            n_parameters = self.n_parameters,
            inits = [1.0] * self.n_parameters,
            bounds = [[0., 10.]] * self.n_parameters,
            auxdata = self.bkg_over_db_squared,
            factors = self.bkg_over_db_squared
        )
        self.suggested_init   = self.constraint.suggested_init
        self.suggested_bounds = self.constraint.suggested_bounds
        self.auxdata = self.constraint.auxdata

    def add_sample(self, channel, sample, modifier_def):
        pass

    def apply(self, channel, sample, pars):
        return pars
