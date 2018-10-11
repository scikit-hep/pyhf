import logging
log = logging.getLogger(__name__)

from . import modifier
from ..paramsets import constrained_by_normal

from .. import get_backend
from ..interpolate import interpolator

@modifier(name='histosys', constrained=True, shared=True, op_code = 'addition')
class histosys(object):
    def __init__(self, nom_data, modifier_data):
        self.n_parameters = 1

        self.at_zero = {}
        self.at_minus_one = {}
        self.at_plus_one = {}


        self.parset = constrained_by_normal(
            n_parameters = self.n_parameters, 
            inits = [0.0],
            bounds = [[-5.,5.]],
            auxdata = [0.]
        )
        assert self.n_parameters == self.parset.n_parameters
        assert self.pdf_type == self.parset.pdf_type

    def add_sample(self, channel, sample, modifier_def):
        log.info('Adding sample {0:s} to channel {1:s}'.format(sample['name'], channel['name']))
        self.at_zero.setdefault(channel['name'], {})[sample['name']] = sample['data']
        self.at_minus_one.setdefault(channel['name'], {})[sample['name']] = modifier_def['data']['lo_data']
        self.at_plus_one.setdefault(channel['name'], {})[sample['name']] = modifier_def['data']['hi_data']

    def apply(self, channel, sample, pars):
        assert int(pars.shape[0]) == 1

        tensorlib, _ = get_backend()
        results = interpolator(0)(
            tensorlib.astensor([
                [
                    [
                        self.at_minus_one[channel['name']][sample['name']],
                        self.at_zero[channel['name']][sample['name']],
                        self.at_plus_one[channel['name']][sample['name']]
                    ]
                ]
            ]), tensorlib.astensor([tensorlib.tolist(pars)])
        )

        return results[0][0][0]
