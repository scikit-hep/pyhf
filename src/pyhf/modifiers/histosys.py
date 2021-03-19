import logging

from .. import get_backend, events
from .. import interpolators
from ..parameters import constrained_by_normal, ParamViewer

log = logging.getLogger(__name__)


def required_parset(sample_data, modifier_data):
    return {
        'paramset_type': constrained_by_normal,
        'n_parameters': 1,
        'is_shared': True,
        'inits': (0.0,),
        'bounds': ((-5.0, 5.0),),
        'fixed': False,
        'auxdata': (0.0,),
    }


class histosys_builder:
    def __init__(self, config):
        self._mega_mods = {}
        self.config = config
        self.required_parsets = {}

    def collect(self, thismod, nom):
        lo_data = thismod['data']['lo_data'] if thismod else nom
        hi_data = thismod['data']['hi_data'] if thismod else nom
        maskval = True if thismod else False
        mask = [maskval] * len(nom)
        return {'lo_data': lo_data, 'hi_data': hi_data, 'mask': mask, 'nom_data': nom}

    def append(self, key, channel, sample, thismod, defined_samp):
        self._mega_mods.setdefault(key, {}).setdefault(sample, {}).setdefault(
            'data', {'hi_data': [], 'lo_data': [], 'nom_data': [], 'mask': []}
        )
        nom = (
            defined_samp['data']
            if defined_samp
            else [0.0] * self.config.channel_nbins[channel]
        )
        moddata = self.collect(thismod, nom)
        self._mega_mods[key][sample]['data']['lo_data'] += moddata['lo_data']
        self._mega_mods[key][sample]['data']['hi_data'] += moddata['hi_data']
        self._mega_mods[key][sample]['data']['nom_data'] += moddata['nom_data']
        self._mega_mods[key][sample]['data']['mask'] += moddata['mask']

        if thismod:
            self.required_parsets.setdefault(thismod['name'], []).append(
                required_parset(defined_samp['data'], thismod['data'])
            )

    def finalize(self):
        return self._mega_mods


class histosys_combined:
    def __init__(
        self, modifiers, pdfconfig, builder_data, interpcode='code0', batch_size=None
    ):
        self.name = 'histosys'
        self.op_code = 'addition'
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
