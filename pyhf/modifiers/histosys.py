from six import with_metaclass
from . import modifier
from .. import tensorlib

@modifier
class histosys(object):
    is_constraint = True

    def __init__(self):
        self.n_parameters = 1
        self.suggested_init = [1.0]
        self.suggested_bounds = [[-5, 5]]

        self.at_zero = {}
        self.at_minus_one = {}
        self.at_plus_one = {}
        self.auxdata = [0]  # observed data is always at a = 1

    def add_sample(self, channel, sample, modifier_data):
        self.at_zero.setdefault(channel['name'], {})[sample['name']] = sample['data']
        self.at_minus_one.setdefault(channel['name'], {})[sample['name']] = modifier_data['lo_data']
        self.at_plus_one.setdefault(channel['name'], {})[sample['name']] = modifier_data['hi_data']

    def alphas(self, pars):
        return pars  # the nuisance parameters correspond directly to the alpha

    def expected_data(self, pars):
        return self.alphas(pars)

    def pdf(self, a, alpha):
        return tensorlib.normal(a, alpha, [1])
