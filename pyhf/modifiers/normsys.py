import logging
log = logging.getLogger(__name__)

from . import modifier
from .. import get_backend
from ..interpolate import interpolator
from ..constraints import standard_gaussian_constraint

@modifier(name='normsys', constrained=True, shared=True, op_code = 'multiplication')
class normsys(object):
    def __init__(self, nom_data, modifier_data):
        self.n_parameters     = 1
        self.suggested_init   = [0.0]
        self.suggested_bounds = [[-5, 5]]
        self.at_zero = [1]
        self.auxdata = [0]  # observed data is always at a = 1

        self.at_minus_one = {}
        self.at_plus_one = {}

        self.constraint = standard_gaussian_constraint(
            n_parameters = self.n_parameters,
            inits = self.suggested_init,
            bounds = self.suggested_bounds,
            auxdata = self.auxdata,
        )

    def add_sample(self, channel, sample, modifier_def):
        log.info('Adding sample {0:s} to channel {1:s}'.format(sample['name'], channel['name']))
        self.at_minus_one.setdefault(channel['name'], {})[sample['name']] = [modifier_def['data']['lo']]
        self.at_plus_one.setdefault(channel['name'], {})[sample['name']]  = [modifier_def['data']['hi']]


    def apply(self, channel, sample, pars):
        # normsysfactor(nom_sys_alphas)   = 1 + sum(interp(1, anchors[i][0], anchors[i][0], val=alpha)  for i in range(nom_sys_alphas))
        assert int(pars.shape[0]) == 1

        tensorlib, _ = get_backend()
        results = interpolator(1)(
            tensorlib.astensor([
                [
                    [
                        self.at_minus_one[channel['name']][sample['name']],
                        self.at_zero,
                        self.at_plus_one[channel['name']][sample['name']]
                    ]
                ]
            ]), tensorlib.astensor([tensorlib.tolist(pars)])
        )

        return results[0][0][0]
