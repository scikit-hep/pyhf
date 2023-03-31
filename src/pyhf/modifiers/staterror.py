import logging
from typing import List

import pyhf
from pyhf import events
from pyhf.exceptions import InvalidModifier
from pyhf.parameters import ParamViewer
from pyhf.tensor.manager import get_backend

log = logging.getLogger(__name__)


def required_parset(sigmas, fixed: List[bool]):
    n_parameters = len(sigmas)
    return {
        'paramset_type': 'constrained_by_normal',
        'n_parameters': n_parameters,
        'is_scalar': False,
        'inits': (1.0,) * n_parameters,
        'bounds': ((1e-10, 10.0),) * n_parameters,
        'fixed': tuple(fixed),
        'auxdata': (1.0,) * n_parameters,
        'sigmas': tuple(sigmas),
    }


class staterror_builder:
    """Builder class for collecting staterror modifier data"""

    is_shared = True

    def __init__(self, config):
        self.builder_data = {}
        self.config = config
        self.required_parsets = {}

    def collect(self, thismod, nom):
        uncrt = thismod['data'] if thismod else [0.0] * len(nom)
        mask = [True if thismod else False] * len(nom)
        return {'mask': mask, 'nom_data': nom, 'uncrt': uncrt}

    def append(self, key, channel, sample, thismod, defined_samp):
        self.builder_data.setdefault(key, {}).setdefault(sample, {}).setdefault(
            'data', {'uncrt': [], 'nom_data': [], 'mask': []}
        )
        nom = (
            defined_samp['data']
            if defined_samp
            else [0.0] * self.config.channel_nbins[channel]
        )
        moddata = self.collect(thismod, nom)
        self.builder_data[key][sample]['data']['mask'].append(moddata['mask'])
        self.builder_data[key][sample]['data']['uncrt'].append(moddata['uncrt'])
        self.builder_data[key][sample]['data']['nom_data'].append(moddata['nom_data'])

    def finalize(self):
        default_backend = pyhf.default_backend

        for modifier_name, modifier in self.builder_data.items():
            for sample_name, sample in modifier.items():
                sample["data"]["mask"] = default_backend.concatenate(
                    sample["data"]["mask"]
                )
                sample["data"]["uncrt"] = default_backend.concatenate(
                    sample["data"]["uncrt"]
                )
                sample["data"]["nom_data"] = default_backend.concatenate(
                    sample["data"]["nom_data"]
                )
                if len(sample["data"]["nom_data"]) != len(sample["data"]["uncrt"]):
                    _modifier_type, _modifier_name = modifier_name.split("/")
                    _sample_data_len = len(sample["data"]["nom_data"])
                    _uncrt_len = len(sample["data"]["uncrt"])
                    raise InvalidModifier(
                        f"The '{sample_name}' sample {_modifier_type} modifier"
                        + f" '{_modifier_name}' has data shape inconsistent with the sample.\n"
                        + f"{sample_name} has 'data' of length {_sample_data_len} but {_modifier_name}"
                        + f" has 'data' of length {_uncrt_len}."
                    )

        for modname in self.builder_data.keys():
            parname = modname.split('/')[1]

            nomsall = default_backend.sum(
                [
                    modifier_data['data']['nom_data']
                    for modifier_data in self.builder_data[modname].values()
                    if default_backend.astensor(modifier_data['data']['mask']).any()
                ],
                axis=0,
            )
            relerrs = default_backend.sum(
                [
                    [
                        (modifier_data['data']['uncrt'][binnr] / nomsall[binnr]) ** 2
                        if nomsall[binnr] > 0
                        else 0.0
                        for binnr in range(len(modifier_data['data']['nom_data']))
                    ]
                    for modifier_data in self.builder_data[modname].values()
                ],
                axis=0,
            )
            # here relerrs still has all the bins, while the staterror are usually per-channel
            # so we need to pick out the masks for this modifier to extract the
            # modifier configuration (sigmas, etc..)
            # so loop over samples and extract the first mask
            # while making sure any subsequent mask is consistent
            relerrs = default_backend.sqrt(relerrs)
            masks = {}
            for modifier_data in self.builder_data[modname].values():
                mask_this_sample = default_backend.astensor(
                    modifier_data['data']['mask'], dtype='bool'
                )
                if mask_this_sample.any():
                    if modname not in masks:
                        masks[modname] = mask_this_sample
                    else:
                        assert (mask_this_sample == masks[modname]).all()

            # extract sigmas using this modifiers mask
            sigmas = relerrs[masks[modname]]

            # list of bools, consistent with other modifiers (no numpy.bool_)
            fixed = default_backend.tolist(sigmas == 0)
            # FIXME: sigmas that are zero will be fixed to 1.0 arbitrarily to ensure
            # non-Nan constraint term, but in a future PR need to remove constraints
            # for these
            sigmas[fixed] = 1.0
            self.required_parsets.setdefault(parname, [required_parset(sigmas, fixed)])
        return self.builder_data


class staterror_combined:
    name = 'staterror'
    op_code = 'multiplication'

    def __init__(self, modifiers, pdfconfig, builder_data, batch_size=None):
        default_backend = pyhf.default_backend
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
        global_concatenated_bin_indices = [
            [[j for c in pdfconfig.channels for j in range(pdfconfig.channel_nbins[c])]]
        ]

        self._access_field = default_backend.tile(
            global_concatenated_bin_indices,
            (len(self._staterr_mods), self.batch_size or 1, 1),
        )

        self._reindex_access_field(pdfconfig)

        self._precompute()
        events.subscribe('tensorlib_changed')(self._precompute)

    def _reindex_access_field(self, pdfconfig):
        default_backend = pyhf.default_backend
        for syst_index, syst_access in enumerate(self._access_field):
            singular_sample_index = [
                idx
                for idx, syst in enumerate(
                    default_backend.astensor(self._staterror_mask)[syst_index, :, 0]
                )
                if any(syst)
            ][-1]

            for batch_index, batch_access in enumerate(syst_access):
                selection = self.param_viewer.index_selection[syst_index][batch_index]
                access_field_for_syst_and_batch = default_backend.zeros(
                    len(batch_access)
                )

                sample_mask = self._staterror_mask[syst_index][singular_sample_index][0]
                access_field_for_syst_and_batch[sample_mask] = selection
                self._access_field[
                    syst_index, batch_index
                ] = access_field_for_syst_and_batch

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
