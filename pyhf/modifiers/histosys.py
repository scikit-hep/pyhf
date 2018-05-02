import logging
log = logging.getLogger(__name__)

from . import modifier
from .. import get_backend
from ..interpolate import interpolator

@modifier(name='histosys', constrained=True, shared=True)
class histosys(object):
    def __init__(self, nom_data, modifier_data):
        self.n_parameters = 1
        self.suggested_init = [0.0]
        self.suggested_bounds = [[-5, 5]]

        self.at_zero = {}
        self.at_minus_one = {}
        self.at_plus_one = {}
        self.auxdata = [0]  # observed data is always at a = 1

    def add_sample(self, channel, sample, modifier_def):
        log.info('Adding sample {0:s} to channel {1:s}'.format(sample['name'], channel['name']))
        self.at_zero.setdefault(channel['name'], {})[sample['name']] = sample['data']
        self.at_minus_one.setdefault(channel['name'], {})[sample['name']] = modifier_def['data']['lo_data']
        self.at_plus_one.setdefault(channel['name'], {})[sample['name']] = modifier_def['data']['hi_data']

    def alphas(self, pars):
        return pars  # the nuisance parameters correspond directly to the alpha

    def expected_data(self, pars):
        return self.alphas(pars)

    def pdf(self, a, alpha):
        tensorlib, _ = get_backend()
        return tensorlib.normal(a, alpha, [1])

    def apply(self, channel, sample, pars):
        assert int(pars.shape[0]) == 1
        return interpolator(0)(self.at_minus_one[channel['name']][sample['name']],
                               self.at_zero[channel['name']][sample['name']],
                               self.at_plus_one[channel['name']][sample['name']],
                               pars)[0]
