import logging

from .. import get_backend, default_backend, events
from ..parameters import constrained_by_normal, ParamViewer

log = logging.getLogger(__name__)


def required_parset(sample_data, modifier_data):
    return {
        'paramset_type': constrained_by_normal,
        'n_parameters': len(sample_data),
        'is_shared': True,
        'inits': (1.0,) * len(sample_data),
        'bounds': ((1e-10, 10.0),) * len(sample_data),
        'fixed': False,
        'auxdata': (1.0,) * len(sample_data),
    }


class staterr_builder:
    def __init__(self, config):
        self._mega_mods = {}
        self.config = config
        self.required_parsets = {}

    def collect(self, thismod, nom):
        uncrt = thismod['data'] if thismod else [0.0] * len(nom)
        mask = [True if thismod else False] * len(nom)
        return {'mask': mask, 'nom_data': nom, 'uncrt': uncrt}

    def append(self, key, channel, sample, thismod, defined_samp):
        self._mega_mods.setdefault(key, {}).setdefault(sample, {}).setdefault(
            'data', {'uncrt': [], 'nom_data': [], 'mask': []}
        )
        nom = (
            defined_samp['data']
            if defined_samp
            else [0.0] * self.config.channel_nbins[channel]
        )
        moddata = self.collect(thismod, nom)
        self._mega_mods[key][sample]['data']['mask'] += moddata['mask']
        self._mega_mods[key][sample]['data']['uncrt'] += moddata['uncrt']
        self._mega_mods[key][sample]['data']['nom_data'] += moddata['nom_data']

        if thismod:
            self.required_parsets.setdefault(thismod['name'], []).append(
                required_parset(defined_samp['data'], thismod['data'])
            )

    def finalize(self):
        return self._mega_mods


class staterror_combined:
    def __init__(self, modifiers, pdfconfig, builder_data, batch_size=None):
        self.name = 'staterror'
        self.op_code = 'multiplication'

        self.batch_size = batch_size

        keys = [f'{mtype}/{m}' for m, mtype in modifiers]
        self._staterr_mods = [m for m, _ in modifiers]

        parfield_shape = (self.batch_size or 1, pdfconfig.npars)
        self.param_viewer = ParamViewer(
            parfield_shape, pdfconfig.par_map, self._staterr_mods
        )

        self._staterror_mask = [
            [[builder_data[m][s]['data']['mask']] for s in pdfconfig.samples]
            for m in keys
        ]
        self.__staterror_uncrt = default_backend.astensor(
            [
                [
                    [
                        builder_data[m][s]['data']['uncrt'],
                        builder_data[m][s]['data']['nom_data'],
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
            global_concatenated_bin_indices,
            (len(self._staterr_mods), self.batch_size or 1, 1),
        )
        # access field is shape (sys, batch, globalbin)
        for s, syst_access in enumerate(self._access_field):
            for t, batch_access in enumerate(syst_access):
                selection = self.param_viewer.index_selection[s][t]
                for b, bin_access in enumerate(batch_access):
                    self._access_field[s, t, b] = (
                        selection[bin_access] if bin_access < len(selection) else 0
                    )

        self._precompute()
        events.subscribe('tensorlib_changed')(self._precompute)

    def _precompute(self):
        if not self.param_viewer.index_selection:
            return
        tensorlib, _ = get_backend()
        self.staterror_mask = tensorlib.astensor(self._staterror_mask, dtype="bool")
        self.staterror_mask = tensorlib.tile(
            self.staterror_mask, (1, 1, self.batch_size or 1, 1)
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
        if not self.param_viewer.index_selection:
            return

        tensorlib, _ = get_backend()
        if self.batch_size is None:
            flat_pars = pars
        else:
            flat_pars = tensorlib.reshape(pars, (-1,))
        statfactors = tensorlib.gather(flat_pars, self.access_field)
        results_staterr = tensorlib.einsum('mab,s->msab', statfactors, self.sample_ones)
        results_staterr = tensorlib.where(
            self.staterror_mask, results_staterr, self.staterror_default
        )
        return results_staterr
