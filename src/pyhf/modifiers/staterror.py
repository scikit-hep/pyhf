import logging

from . import modifier
from ..paramsets import constrained_by_normal
from .. import get_backend, default_backend, events
from ..paramview import ParamViewer

log = logging.getLogger(__name__)


@modifier(name='staterror', constrained=True, op_code='multiplication')
class staterror(object):
    @classmethod
    def required_parset(cls, n_parameters):
        return {
            'paramset_type': constrained_by_normal,
            'n_parameters': n_parameters,
            'modifier': cls.__name__,
            'is_constrained': cls.is_constrained,
            'is_shared': True,
            'inits': (1.0,) * n_parameters,
            'bounds': ((1e-10, 10.0),) * n_parameters,
            'auxdata': (1.0,) * n_parameters,
        }


class staterror_combined(object):
    def __init__(self, staterr_mods, pdfconfig, mega_mods, batch_size=1):
        self.batch_size = batch_size

        pnames = [pname for pname, _ in staterr_mods]
        keys = ['{}/{}'.format(mtype, m) for m, mtype in staterr_mods]
        staterr_mods = [m for m, _ in staterr_mods]

        parfield_shape = (self.batch_size, len(pdfconfig.suggested_init()))
        self.parameters_helper = ParamViewer(
            parfield_shape, pdfconfig.par_map, pnames, regular=False
        )

        self._staterr_mods = staterr_mods
        self._staterror_mask = [
            [[mega_mods[s][m]['data']['mask']] for s in pdfconfig.samples] for m in keys
        ]
        self.__staterror_uncrt = default_backend.astensor(
            [
                [
                    [
                        mega_mods[s][m]['data']['uncrt'],
                        mega_mods[s][m]['data']['nom_data'],
                    ]
                    for s in pdfconfig.samples
                ]
                for m in keys
            ]
        )
        self.finalize(pdfconfig)

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
        self.staterror_mask = tensorlib.astensor(self._staterror_mask)
        self.staterror_mask = tensorlib.tile(
            self.staterror_mask, (1, 1, self.batch_size, 1)
        )
        self.access_field = tensorlib.astensor(self._access_field, dtype='int')
        self.sample_ones = tensorlib.ones(tensorlib.shape(self.staterror_mask)[1])
        self.staterror_default = tensorlib.ones(tensorlib.shape(self.staterror_mask))

    def finalize(self, pdfconfig):
        staterror_mask = default_backend.astensor(self._staterror_mask)
        for this_mask, uncert_this_mod, mod in zip(
            staterror_mask, self.__staterror_uncrt, self._staterr_mods
        ):
            active_nominals = default_backend.where(
                this_mask[:, 0, :],
                uncert_this_mod[:, 1, :],
                default_backend.zeros(uncert_this_mod[:, 1, :].shape),
            )
            summed_nominals = default_backend.sum(active_nominals, axis=0)

            # the below tries to filter cases in which this modifier is not
            # used by checking non zeroness.. should probably use mask
            numerator = default_backend.where(
                uncert_this_mod[:, 1, :] > 0,
                uncert_this_mod[:, 0, :],
                default_backend.zeros(uncert_this_mod[:, 1, :].shape),
            )
            denominator = default_backend.where(
                summed_nominals > 0,
                summed_nominals,
                default_backend.ones(uncert_this_mod[:, 1, :].shape),
            )
            relerrs = numerator / denominator
            sigmas = default_backend.sqrt(
                default_backend.sum(default_backend.power(relerrs, 2), axis=0)
            )
            assert len(sigmas[sigmas > 0]) == pdfconfig.param_set(mod).n_parameters
            pdfconfig.param_set(mod).sigmas = default_backend.tolist(sigmas[sigmas > 0])

    def apply(self, pars):
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
        statfactors = tensorlib.gather(flat_pars, self.access_field)
        results_staterr = tensorlib.einsum('yab,s->ysab', statfactors, self.sample_ones)
        results_staterr = tensorlib.where(
            self.staterror_mask, results_staterr, self.staterror_default
        )
        return results_staterr
