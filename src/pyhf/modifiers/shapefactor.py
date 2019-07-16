import logging

from . import modifier
from ..paramsets import unconstrained
from .. import get_backend, default_backend, events

log = logging.getLogger(__name__)


@modifier(name='shapefactor', op_code='multiplication')
class shapefactor(object):
    @classmethod
    def required_parset(cls, n_parameters):
        return {
            'paramset_type': unconstrained,
            'n_parameters': n_parameters,
            'modifier': cls.__name__,
            'is_constrained': cls.is_constrained,
            'is_shared': True,
            'inits': (1.0,) * n_parameters,
            'bounds': ((0.0, 10.0),) * n_parameters,
        }


class shapefactor_combined(object):
    def __init__(self, shapefactor_mods, pdfconfig, mega_mods):
        """
        Imagine a situation where we have 2 channels (SR, CR), 3 samples (sig1,
        bkg1, bkg2), and 2 shapefactor modifiers (coupled_shapefactor,
        uncoupled_shapefactor). Let's say this is the set-up:

            SR(nbins=2)
              sig1 -> subscribes to normfactor
              bkg1 -> subscribes to coupled_shapefactor
            CR(nbins=3)
              bkg2 -> subscribes to coupled_shapefactor, uncoupled_shapefactor

        The coupled_shapefactor needs to have 3 nuisance parameters to account
        for the CR, with 2 of them shared in the SR. The uncoupled_shapefactor
        just has 3 nuisance parameters.

        self._parindices will look like
            [0, 1, 2, 3, 4, 5, 6]

        self._shapefactor_indices will look like
            [[1,2,3],[4,5,6]]
             ^^^^^^^         = coupled_shapefactor
                     ^^^^^^^ = uncoupled_shapefactor

        with the 0th par-index corresponding to the normfactor. Because
        channel1 has 2 bins, and channel2 has 3 bins (with channel1 before
        channel2), global_concatenated_bin_indices looks like
            [0, 1, 0, 1, 2]
            ^^^^^            = channel1
                  ^^^^^^^^^  = channel2

        So now we need to gather the corresponding shapefactor indices
        according to global_concatenated_bin_indices. Therefore
        self._shapefactor_indices now looks like
            [[1, 2, 1, 2, 3], [4, 5, 4, 5, 6]]

        and at that point can be used to compute the effect of shapefactor.
        """

        pnames = [pname for pname, _ in shapefactor_mods]
        keys = ['{}/{}'.format(mtype, m) for m, mtype in shapefactor_mods]
        shapefactor_mods = [m for m, _ in shapefactor_mods]

        self._parindices = list(range(len(pdfconfig.suggested_init())))
        self._shapefactor_indices = [
            self._parindices[pdfconfig.par_slice(p)] for p in pnames
        ]
        self._shapefactor_mask = [
            [[mega_mods[s][m]['data']['mask']] for s in pdfconfig.samples] for m in keys
        ]
        global_concatenated_bin_indices = [
            j for c in pdfconfig.channels for j in range(pdfconfig.channel_nbins[c])
        ]
        # compute the max so that we can pad with 0s for the right shape
        # for gather. The 0s will get masked by self._shapefactor_mask anyway
        # For example: [[1,2,3],[4,5],[6,7,8]] -> [[1,2,3],[4,5,0],[6,7,8]]
        max_nbins = max(global_concatenated_bin_indices) + 1
        self._shapefactor_indices = [
            default_backend.tolist(
                default_backend.gather(
                    default_backend.astensor(
                        indices + [0] * (max_nbins - len(indices)), dtype='int'
                    ),
                    global_concatenated_bin_indices,
                )
            )
            for indices in self._shapefactor_indices
        ]

        self._precompute()
        events.subscribe('tensorlib_changed')(self._precompute)

    def _precompute(self):
        if not self._shapefactor_indices:
            return
        tensorlib, _ = get_backend()
        self.shapefactor_mask = tensorlib.astensor(self._shapefactor_mask)
        self.shapefactor_default = tensorlib.ones(
            tensorlib.shape(self.shapefactor_mask)
        )
        self.shapefactor_indices = tensorlib.astensor(
            self._shapefactor_indices, dtype='int'
        )
        self.sample_ones = tensorlib.ones(tensorlib.shape(self.shapefactor_mask)[1])
        self.alpha_ones = tensorlib.ones([1])

    def apply(self, pars):
        if not self._shapefactor_indices:
            return
        tensorlib, _ = get_backend()
        shapefactors = tensorlib.gather(pars, self.shapefactor_indices)
        results_shapefactor = tensorlib.einsum(
            's,a,mb->msab', self.sample_ones, self.alpha_ones, shapefactors
        )
        results_shapefactor = tensorlib.where(
            self.shapefactor_mask, results_shapefactor, self.shapefactor_default
        )
        return results_shapefactor
