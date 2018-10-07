import logging
log = logging.getLogger(__name__)

from . import modifier
from ..paramsets import constrained_by_normal

from .. import get_backend, default_backend

@modifier(name='staterror', shared=True, constrained=True, op_code = 'multiplication')
class staterror(object):
    def __init__(self, nom_data, modifier_data):
        self.n_parameters = len(nom_data)
        self.nominal_counts   = []
        self.uncertainties    = []
        self.channel = None

        self.parset = constrained_by_normal(
            n_parameters = self.n_parameters,
            inits = [1.] * self.n_parameters,
            bounds = [[0., 10.]] * self.n_parameters,
            auxdata = [1.] * self.n_parameters
        )
        assert self.n_parameters == self.parset.n_parameters
        assert self.pdf_type == self.parset.pdf_type

    def finalize(self):
        # this computes sum_i uncertainty_i for all samples
        # (broadcastted for all bins in the channel)
        # for each bin, the relative uncert is the width of a gaussian
        # which is the constraint pdf; Prod_i Gaus(x = a_i, mu = alpha_i, sigma = relunc_i)
        inquad = default_backend.sqrt(default_backend.sum(default_backend.power(self.uncertainties,2), axis=0))
        totals = default_backend.sum(self.nominal_counts,axis=0)
        self.parset.sigmas = default_backend.tolist(default_backend.divide(inquad,totals))

    def add_sample(self, channel, sample, modifier_def):
        self.nominal_counts.append(sample['data'])
        self.uncertainties.append(modifier_def['data'])
        if self.channel and self.channel != channel['name']:
            raise RuntimeError('not sure yet how to deal with this case')
        self.channel = channel['name']

    def apply(self, channel, sample, pars):
        raise RuntimeError()
