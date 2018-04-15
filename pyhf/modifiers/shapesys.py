from six import with_metaclass
from . import IModifier
from .. import tensorlib

class shapesys(with_metaclass(IModifier, object)):
    is_constraint = True

    @staticmethod
    def suggested_init(n_parameters):
        return [1.0] * n_parameters

    @staticmethod
    def suggested_bounds(n_parameters):
        return [[0, 10]] * n_parameters

    def __init__(self, nom_data, modifier_data):
        self.auxdata = []
        self.bkg_over_db_squared = []
        for b, deltab in zip(nom_data, modifier_data):
            bkg_over_bsq = b * b / deltab / deltab  # tau*b
            log.info('shapesys for b,delta b (%s, %s) -> tau*b = %s',
                     b, deltab, bkg_over_bsq)
            self.bkg_over_db_squared.append(bkg_over_bsq)
            self.auxdata.append(bkg_over_bsq)

    def alphas(self, pars):
        return tensorlib.product(tensorlib.stack([pars, tensorlib.astensor(self.bkg_over_db_squared)]), axis=0)

    def pdf(self, a, alpha):
        return tensorlib.poisson(a, alpha)

    def expected_data(self, pars):
        return self.alphas(pars)
