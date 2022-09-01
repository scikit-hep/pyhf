import logging

import pyhf
from pyhf import events
from pyhf.exceptions import InvalidModifier
from pyhf.parameters import ParamViewer
from pyhf.tensor.manager import get_backend

log = logging.getLogger(__name__)


def required_parset(sample_data, modifier_data):
    # count the number of bins with nonzero, positive yields
    valid_bins = [
        (sample_bin > 0 and modifier_bin > 0)
        for sample_bin, modifier_bin in zip(modifier_data, sample_data)
    ]

    factors = [
        (nom_yield**2 / unc**2) if (is_valid) else 1.0
        for is_valid, nom_yield, unc in zip(valid_bins, sample_data, modifier_data)
    ]
    fixed = tuple(not is_valid for is_valid in valid_bins)
    n_parameters = len(factors)
    return {
        "paramset_type": "constrained_by_poisson",
        "n_parameters": n_parameters,
        "is_scalar": False,
        "inits": (1.0,) * n_parameters,
        "bounds": ((1e-10, 10.0),) * n_parameters,
        "fixed": fixed,
        "auxdata": tuple(factors),
        "factors": tuple(factors),
    }


class shapesys_builder:
    """Builder class for collecting shapesys modifier data"""

    is_shared = False

    def __init__(self, config):
        self.builder_data = {}
        self.config = config
        self.required_parsets = {}

    def collect(self, thismod, nom):
        uncrt = thismod['data'] if thismod else [0.0] * len(nom)
        mask = [True] * len(nom) if thismod else [False] * len(nom)
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

        if thismod:
            self.required_parsets.setdefault(
                thismod['name'],
                [required_parset(defined_samp['data'], thismod['data'])],
            )

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
        return self.builder_data


class shapesys_combined:
    name = 'shapesys'
    op_code = 'multiplication'

    def __init__(self, modifiers, pdfconfig, builder_data, batch_size=None):
        default_backend = pyhf.default_backend
        self.batch_size = batch_size

        keys = [f'{mtype}/{m}' for m, mtype in modifiers]
        self._shapesys_mods = [m for m, _ in modifiers]

        parfield_shape = (self.batch_size or 1, pdfconfig.npars)
        self.param_viewer = ParamViewer(
            parfield_shape, pdfconfig.par_map, self._shapesys_mods
        )

        self._shapesys_mask = [
            [[builder_data[m][s]['data']['mask']] for s in pdfconfig.samples]
            for m in keys
        ]
        self.__shapesys_info = default_backend.astensor(
            [
                [
                    [
                        builder_data[m][s]['data']['mask'],
                        builder_data[m][s]['data']['nom_data'],
                        builder_data[m][s]['data']['uncrt'],
                    ]
                    for s in pdfconfig.samples
                ]
                for m in keys
            ]
        )
        global_concatenated_bin_indices = [
            [[j for c in pdfconfig.channels for j in range(pdfconfig.channel_nbins[c])]]
        ]

        self._access_field = default_backend.tile(
            global_concatenated_bin_indices,
            (len(self._shapesys_mods), self.batch_size or 1, 1),
        )
        # access field is shape (sys, batch, globalbin)

        # reindex it based on current masking
        self._reindex_access_field(pdfconfig)

        self._precompute()
        events.subscribe('tensorlib_changed')(self._precompute)

    def _reindex_access_field(self, pdfconfig):
        default_backend = pyhf.default_backend

        for syst_index, syst_access in enumerate(self._access_field):
            singular_sample_index = [
                idx
                for idx, syst in enumerate(
                    default_backend.astensor(self._shapesys_mask)[syst_index, :, 0]
                )
                if any(syst)
            ][-1]

            for batch_index, batch_access in enumerate(syst_access):
                selection = self.param_viewer.index_selection[syst_index][batch_index]
                access_field_for_syst_and_batch = default_backend.zeros(
                    len(batch_access)
                )

                sample_mask = self._shapesys_mask[syst_index][singular_sample_index][0]
                access_field_for_syst_and_batch[sample_mask] = selection
                self._access_field[
                    syst_index, batch_index
                ] = access_field_for_syst_and_batch

    def _precompute(self):
        tensorlib, _ = get_backend()
        if not self.param_viewer.index_selection:
            return
        self.shapesys_mask = tensorlib.astensor(self._shapesys_mask, dtype="bool")
        self.shapesys_mask = tensorlib.tile(
            self.shapesys_mask, (1, 1, self.batch_size or 1, 1)
        )
        self.access_field = tensorlib.astensor(self._access_field, dtype='int')
        self.sample_ones = tensorlib.ones(tensorlib.shape(self.shapesys_mask)[1])
        self.shapesys_default = tensorlib.ones(tensorlib.shape(self.shapesys_mask))

    def apply(self, pars):
        """
        Returns:
            modification tensor: Shape (n_modifiers, n_global_samples, n_alphas, n_global_bin)
        """
        tensorlib, _ = get_backend()
        if not self.param_viewer.index_selection:
            return
        tensorlib, _ = get_backend()
        if self.batch_size is None:
            flat_pars = pars
        else:
            flat_pars = tensorlib.reshape(pars, (-1,))
        shapefactors = tensorlib.gather(flat_pars, self.access_field)
        results_shapesys = tensorlib.einsum(
            'mab,s->msab', shapefactors, self.sample_ones
        )

        results_shapesys = tensorlib.where(
            self.shapesys_mask, results_shapesys, self.shapesys_default
        )
        return results_shapesys
