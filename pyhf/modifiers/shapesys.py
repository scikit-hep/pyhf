import logging
log = logging.getLogger(__name__)

from . import modifier
from .. import get_backend

@modifier(name='shapesys', constrained=True)
class shapesys(object):
    def __init__(self, nom_data, modifier_data):
        self.n_parameters = len(nom_data)
        self.suggested_init = [1.0] * self.n_parameters
        self.suggested_bounds = [[0, 10]] * self.n_parameters

        self.auxdata = []
        self.bkg_over_db_squared = []
        for b, deltab in zip(nom_data, modifier_data):
            bkg_over_bsq = b * b / deltab / deltab  # tau*b
            log.info('shapesys for b,delta b (%s, %s) -> tau*b = %s',
                     b, deltab, bkg_over_bsq)
            self.bkg_over_db_squared.append(bkg_over_bsq)
            self.auxdata.append(bkg_over_bsq)

    def alphas(self, pars):
        tensorlib, _ = get_backend()
        return tensorlib.product(tensorlib.stack([pars, tensorlib.astensor(self.bkg_over_db_squared)]), axis=0)

    def pdf(self, a, alpha):
        tensorlib, _ = get_backend()
        return tensorlib.poisson(a, alpha)

    def expected_data(self, pars):
        return self.alphas(pars)

    def add_sample(self, channel, sample, modifier_def):
        pass
