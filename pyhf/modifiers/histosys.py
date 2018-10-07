import logging
log = logging.getLogger(__name__)

from . import modifier
from .. import get_backend
from ..interpolate import interpolator

@modifier(name='histosys', constrained=True, shared=True, op_code = 'addition')
class histosys(object):
    """HistoSys modifier

        >>> import pyhf
        >>> pyhf.set_backend(pyhf.default_backend)
        >>> mymod = pyhf.modifiers.histosys([1.0],[0.0])
        >>> mymod.alphas([1.0,2.0,3.0])
        [1.0, 2.0, 3.0]
    """
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
