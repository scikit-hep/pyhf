import logging

import pyhf
from pyhf import events, interpolators
from pyhf.exceptions import InvalidModifier
from pyhf.parameters import ParamViewer
from pyhf.tensor.manager import get_backend

log = logging.getLogger(__name__)


def required_parset(sample_data, modifier_data):
    return {
        'paramset_type': 'constrained_by_normal',
        'n_parameters': 1,
        'is_shared': True,
        'is_scalar': True,
        'inits': (0.0,),
        'bounds': ((-5.0, 5.0),),
        'fixed': False,
        'auxdata': (0.0,),
    }


class histosys_builder:
    """Builder class for collecting histoys modifier data"""

    def __init__(self, config):
        self.builder_data = {}
        self.config = config
        self.required_parsets = {}

    def collect(self, thismod, nom):
        lo_data = thismod['data']['lo_data'] if thismod else nom
        hi_data = thismod['data']['hi_data'] if thismod else nom
        maskval = bool(thismod)
        mask = [maskval] * len(nom)
        return {'lo_data': lo_data, 'hi_data': hi_data, 'mask': mask, 'nom_data': nom}

    def append(self, key, channel, sample, thismod, defined_samp):
        self.builder_data.setdefault(key, {}).setdefault(sample, {}).setdefault(
            'data', {'hi_data': [], 'lo_data': [], 'nom_data': [], 'mask': []}
        )
        nom = (
            defined_samp['data']
            if defined_samp
            else [0.0] * self.config.channel_nbins[channel]
        )
        moddata = self.collect(thismod, nom)
        self.builder_data[key][sample]['data']['lo_data'].append(moddata['lo_data'])
        self.builder_data[key][sample]['data']['hi_data'].append(moddata['hi_data'])
        self.builder_data[key][sample]['data']['nom_data'].append(moddata['nom_data'])
        self.builder_data[key][sample]['data']['mask'].append(moddata['mask'])

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
                sample["data"]["lo_data"] = default_backend.concatenate(
                    sample["data"]["lo_data"]
                )
                sample["data"]["hi_data"] = default_backend.concatenate(
                    sample["data"]["hi_data"]
                )
                sample["data"]["nom_data"] = default_backend.concatenate(
                    sample["data"]["nom_data"]
                )
                if (
                    not len(sample["data"]["nom_data"])
                    == len(sample["data"]["lo_data"])
                    == len(sample["data"]["hi_data"])
                ):
                    _modifier_type, _modifier_name = modifier_name.split("/")
                    _sample_data_len = len(sample["data"]["nom_data"])
                    _lo_data_len = len(sample["data"]["lo_data"])
                    _hi_data_len = len(sample["data"]["hi_data"])
                    raise InvalidModifier(
                        f"The '{sample_name}' sample {_modifier_type} modifier"
                        + f" '{_modifier_name}' has data shape inconsistent with the sample.\n"
                        + f"{sample_name} has 'data' of length {_sample_data_len} but {_modifier_name}"
                        + f" has 'lo_data' of length {_lo_data_len} and 'hi_data' of length {_hi_data_len}."
                    )
        return self.builder_data


class histosys_combined:
    name = 'histosys'
    op_code = 'addition'

    def __init__(
        self, modifiers, pdfconfig, builder_data, interpcode='code0', batch_size=None
    ):
        self.batch_size = batch_size
        self.interpcode = interpcode
        assert self.interpcode in ['code0', 'code2', 'code4p']

        keys = [f'{mtype}/{m}' for m, mtype in modifiers]
        histosys_mods = [m for m, _ in modifiers]

        parfield_shape = (
            (self.batch_size, pdfconfig.npars)
            if self.batch_size
            else (pdfconfig.npars,)
        )
        self.param_viewer = ParamViewer(
            parfield_shape, pdfconfig.par_map, histosys_mods
        )

        self._histosys_histoset = [
            [
                [
                    builder_data[m][s]['data']['lo_data'],
                    builder_data[m][s]['data']['nom_data'],
                    builder_data[m][s]['data']['hi_data'],
                ]
                for s in pdfconfig.samples
            ]
            for m in keys
        ]
        self._histosys_mask = [
            [[builder_data[m][s]['data']['mask']] for s in pdfconfig.samples]
            for m in keys
        ]

        if histosys_mods:
            self.interpolator = getattr(interpolators, self.interpcode)(
                self._histosys_histoset
            )

        self._precompute()
        events.subscribe('tensorlib_changed')(self._precompute)

    def _precompute(self):
        if not self.param_viewer.index_selection:
            return
        tensorlib, _ = get_backend()
        self.histosys_mask = tensorlib.astensor(self._histosys_mask, dtype="bool")
        self.histosys_default = tensorlib.zeros(self.histosys_mask.shape)
        if self.batch_size is None:
            self.indices = tensorlib.reshape(
                self.param_viewer.indices_concatenated, (-1, 1)
            )

    def apply(self, pars):
        """
        Returns:
            modification tensor: Shape (n_modifiers, n_global_samples, n_alphas, n_global_bin)
        """
        if not self.param_viewer.index_selection:
            return

        tensorlib, _ = get_backend()
        if self.batch_size is None:
            histosys_alphaset = self.param_viewer.get(pars, self.indices)
        else:
            histosys_alphaset = self.param_viewer.get(pars)

        results_histo = self.interpolator(histosys_alphaset)
        # either rely on numerical no-op or force with line below
        results_histo = tensorlib.where(
            self.histosys_mask, results_histo, self.histosys_default
        )
        return results_histo
