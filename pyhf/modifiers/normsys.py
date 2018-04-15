import logging
log = logging.getLogger(__name__)

from six import with_metaclass
from . import IModifier
from .. import tensorlib

class normsys(with_metaclass(IModifier, object)):
    is_constrained = True

    def __init__(self):
        self.n_parameters = 1
        self.suggested_init = [0.0]
        self.suggested_bounds = [[-5, 5]]

        self.at_zero = 1
        self.at_minus_one = {}
        self.at_plus_one = {}
        self.auxdata = [0]  # observed data is always at a = 1

    def add_sample(self, channel, sample, modifier_data):
        self.at_minus_one.setdefault(channel['name'], {})[sample['name']] = modifier_data['lo']
        self.at_plus_one.setdefault(channel['name'], {})[sample['name']] = modifier_data['hi']

    def alphas(self, pars):
        return pars  # the nuisance parameters correspond directly to the alpha

    def expected_data(self, pars):
        return self.alphas(pars)

    def pdf(self, a, alpha):
        return tensorlib.normal(a, alpha, 1)
