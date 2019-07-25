import logging

from . import modifier
from ..paramsets import unconstrained
from .. import get_backend, default_backend, events
from ..paramview import ParamViewer

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
    def __init__(self, shapefactor_mods, pdfconfig, mega_mods, batch_size=1):
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

        self.batch_size = batch_size
        pnames = [pname for pname, _ in shapefactor_mods]
        keys = ['{}/{}'.format(mtype, m) for m, mtype in shapefactor_mods]
        shapefactor_mods = [m for m, _ in shapefactor_mods]

        parfield_shape = (self.batch_size, len(pdfconfig.suggested_init()))
        self.parameters_helper = ParamViewer(
            parfield_shape, pdfconfig.par_map, pnames, regular=False
        )

        self._shapefactor_mask = [
            [[mega_mods[s][m]['data']['mask']] for s in pdfconfig.samples] for m in keys
        ]

        global_concatenated_bin_indices = [
            [[j for c in pdfconfig.channels for j in range(pdfconfig.channel_nbins[c])]]
        ]

        self._access_field = default_backend.tile(
            global_concatenated_bin_indices, (len(pnames), self.batch_size, 1)
        )
        # access field is shape (sys, batch, globalbin)
        for s, syst_access in enumerate(self._access_field):
            for t, batch_access in enumerate(syst_access):
                selection = self.parameters_helper.index_selection[s][t]
                for b, bin_access in enumerate(batch_access):
                    self._access_field[s, t, b] = (
                        selection[bin_access] if bin_access < len(selection) else 0
                    )

        self._precompute()
        events.subscribe('tensorlib_changed')(self._precompute)

    def _precompute(self):
        if not self.parameters_helper.index_selection:
            return
        tensorlib, _ = get_backend()
        self.shapefactor_mask = tensorlib.astensor(self._shapefactor_mask)
        self.shapefactor_mask = tensorlib.tile(
            self.shapefactor_mask, (1, 1, self.batch_size, 1)
        )
        self.access_field = tensorlib.astensor(self._access_field, dtype='int')

        self.shapefactor_default = tensorlib.ones(
            tensorlib.shape(self.shapefactor_mask)
        )
        self.sample_ones = tensorlib.ones(tensorlib.shape(self.shapefactor_mask)[1])

    def apply(self, pars):
        '''
        Returns:
            modification tensor: Shape (n_modifiers, n_global_samples, n_alphas, n_global_bin)
        '''
        if not self.parameters_helper.index_selection:
            return

        tensorlib, _ = get_backend()
        pars = tensorlib.astensor(pars)
        if self.batch_size == 1:
            batched_pars = tensorlib.reshape(
                pars, (self.batch_size,) + tensorlib.shape(pars)
            )
        else:
            batched_pars = pars

        flat_pars = tensorlib.reshape(batched_pars, (-1,))
        shapefactors = tensorlib.gather(flat_pars, self.access_field)
        results_shapefactor = tensorlib.einsum(
            'yab,s->ysab', shapefactors, self.sample_ones
        )
        results_shapefactor = tensorlib.where(
            self.shapefactor_mask, results_shapefactor, self.shapefactor_default
        )
        return results_shapefactor
