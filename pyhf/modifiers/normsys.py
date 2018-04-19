import logging
log = logging.getLogger(__name__)

from . import modifier
from .. import get_backend

@modifier(name='normsys', constrained=True, shared=True)
class normsys(object):
    def __init__(self, nom_data, modifier_data):
        self.n_parameters = 1
        self.suggested_init = [0.0]
        self.suggested_bounds = [[-5, 5]]

        self.at_zero = 1
        self.at_minus_one = {}
        self.at_plus_one = {}
        self.auxdata = [0]  # observed data is always at a = 1

    def add_sample(self, channel, sample, modifier_def):
        log.info('Adding sample {0:s} to channel {1:s}'.format(sample['name'], channel['name']))
        self.at_minus_one.setdefault(channel['name'], {})[sample['name']] = modifier_def['data']['lo']
        self.at_plus_one.setdefault(channel['name'], {})[sample['name']] = modifier_def['data']['hi']

    def alphas(self, pars):
        return pars  # the nuisance parameters correspond directly to the alpha

    def expected_data(self, pars):
        return self.alphas(pars)

    def pdf(self, a, alpha):
        tensorlib, _ = get_backend()
        return tensorlib.normal(a, alpha, 1)

    def apply(self, channel, sample, pars):
        assert int(pars.shape[0]) == 1
        return self._apply(self.at_minus_one[channel['name']][sample['name']],
                           self.at_zero,
                           self.at_plus_one[channel['name']][sample['name']],
                           pars)[0]

    @staticmethod
    def _apply(at_minus_one, at_zero, at_plus_one, alphas):
        tensorlib, _ = get_backend()
        at_minus_one = tensorlib.astensor(at_minus_one)
        at_zero = tensorlib.astensor(at_zero)
        at_plus_one = tensorlib.astensor(at_plus_one)
        alphas = tensorlib.astensor(alphas)

        base_positive = tensorlib.divide(at_plus_one,  at_zero)
        base_negative = tensorlib.divide(at_minus_one, at_zero)
        expo_positive = tensorlib.outer(alphas, tensorlib.ones(base_positive.shape))
        mask = tensorlib.outer(alphas > 0, tensorlib.ones(base_positive.shape))
        bases = tensorlib.where(mask,base_positive,base_negative)
        exponents = tensorlib.where(mask, expo_positive,-expo_positive)
        return tensorlib.power(bases, exponents)
