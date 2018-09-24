import logging
log = logging.getLogger(__name__)

from . import modifier
from ..paramsets import constrained_by_poisson

@modifier(name='shapesys', constrained=True, pdf_type='poisson', op_code = 'multiplication')
class shapesys(object):
    def __init__(self, nom_data, modifier_data):
        self.n_parameters = len(nom_data)
        self.channel = None

        bkg_over_db_squared = []
        for b, deltab in zip(nom_data, modifier_data):
            bkg_over_bsq = b * b / deltab / deltab  # tau*b
            log.info('shapesys for b,delta b (%s, %s) -> tau*b = %s',
                     b, deltab, bkg_over_bsq)
            bkg_over_db_squared.append(bkg_over_bsq)

        self.parset = constrained_by_poisson(
            n_parameters = self.n_parameters,
            inits = [1.0] * self.n_parameters,
            bounds = [[0., 10.]] * self.n_parameters,
            auxdata = bkg_over_db_squared,
            factors = bkg_over_db_squared
        )

        assert self.n_parameters == self.parset.n_parameters
        assert self.pdf_type == self.parset.pdf_type

    def add_sample(self, channel, sample, modifier_def):
        if self.channel and self.channel != channel['name']:
            raise RuntimeError('not sure yet how to deal with this case')
        self.channel = channel['name']

    def apply(self, channel, sample, pars):
        return pars
