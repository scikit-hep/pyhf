import logging
log = logging.getLogger(__name__)

from . import modifier
from .. import get_backend, default_backend
from ..constraints import standard_gaussian_constraint

@modifier(name='staterror', shared=True, constrained=True, op_code = 'multiplication')
class staterror(object):
    def __init__(self, nom_data, modifier_data):
        self.n_parameters     = len(nom_data)
        self.suggested_init   = [1.0] * self.n_parameters
        self.suggested_bounds = [[0, 10]] * self.n_parameters
        self.auxdata          = [1.] * self.n_parameters
        self.nominal_counts   = []
        self.uncertainties    = []

        self.constraint = standard_gaussian_constraint(
            n_parameters = self.n_parameters,
            inits = self.suggested_init,
            bounds = self.suggested_bounds,
            auxdata = self.auxdata,
        )

    def finalize(self):
        tensorlib, _ = get_backend()
        # this computes sum_i uncertainty_i for all samples
        # (broadcastted for all bins in the channel)
        # for each bin, the relative uncert is the width of a gaussian
        # which is the constraint pdf; Prod_i Gaus(x = a_i, mu = alpha_i, sigma = relunc_i)
        inquad = default_backend.sqrt(default_backend.sum(default_backend.power(self.uncertainties,2), axis=0))
        totals = default_backend.sum(self.nominal_counts,axis=0)
        self.sigmas = default_backend.tolist(default_backend.divide(inquad,totals))

    def add_sample(self, channel, sample, modifier_def):
        self.nominal_counts.append(sample['data'])
        self.uncertainties.append(modifier_def['data'])

    def apply(self, channel, sample, pars):
        return pars
