import logging
log = logging.getLogger(__name__)

from . import modifier
from .. import get_backend

@modifier(name='staterror', shared=True, constrained=True)
class staterror(object):
    def __init__(self, nom_data, modifier_data):
        self.n_parameters     = len(nom_data)
        self.suggested_init   = [1.0] * self.n_parameters
        self.suggested_bounds = [[0, 10]] * self.n_parameters
        self.auxdata          = [1.] * self.n_parameters
        self.nominal_counts   = []
        self.uncertainties    = []


    def alphas(self, pars):
        return pars  # nuisance parameters are also the means of the

    def pdf(self, a, alpha):
        tensorlib, _ = get_backend()
        # this computes sum_i uncertainty_i for all samples
        # (broadcastted for all bins in the channel)
        # for each bin, the relative uncert is the width of a gaussian
        # which is the constraint pdf; Prod_i Gaus(x = a_i, mu = alpha_i, sigma = relunc_i)
        inquad = tensorlib.sqrt(tensorlib.sum(tensorlib.power(self.uncertainties,2), axis=0))
        totals = tensorlib.sum(self.nominal_counts,axis=0)
        uncrts = tensorlib.divide(inquad,totals)
        return getattr(tensorlib, self.pdf_type)(a, alpha, uncrts)

    def expected_data(self, pars):
        return self.alphas(pars)

    def add_sample(self, channel, sample, modifier_def):
        self.nominal_counts.append(sample['data'])
        self.uncertainties.append(modifier_def['data'])

    def apply(self, channel, sample, pars):
        return pars
