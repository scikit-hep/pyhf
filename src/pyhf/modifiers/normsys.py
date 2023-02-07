import logging

from pyhf import get_backend, events
from pyhf import interpolators
from pyhf.parameters import ParamViewer

log = logging.getLogger(__name__)


def required_parset(sample_data, modifier_data):
    return {
        'paramset_type': 'constrained_by_normal',
        'n_parameters': 1,
        'is_scalar': True,
        'inits': (0.0,),
        'bounds': ((-5.0, 5.0),),
        'fixed': False,
        'auxdata': (0.0,),
    }


class normsys_builder:
    """Builder class for collecting normsys modifier data"""

    is_shared = True

    def __init__(self, config):
        self.builder_data = {}
        self.config = config
        self.required_parsets = {}

    def collect(self, thismod, nom):
        maskval = True if thismod else False
        lo_factor = thismod['data']['lo'] if thismod else 1.0
        hi_factor = thismod['data']['hi'] if thismod else 1.0
        nom_data = [1.0] * len(nom)
        lo = [lo_factor] * len(nom)  # broadcasting
        hi = [hi_factor] * len(nom)
        mask = [maskval] * len(nom)
        return {'lo': lo, 'hi': hi, 'mask': mask, 'nom_data': nom_data}

    def append(self, key, channel, sample, thismod, defined_samp):
        self.builder_data.setdefault(key, {}).setdefault(sample, {}).setdefault(
            'data', {'hi': [], 'lo': [], 'nom_data': [], 'mask': []}
        )

        nom = (
            defined_samp['data']
            if defined_samp
            else [0.0] * self.config.channel_nbins[channel]
        )
        moddata = self.collect(thismod, nom)
        self.builder_data[key][sample]['data']['nom_data'] += moddata['nom_data']
        self.builder_data[key][sample]['data']['lo'] += moddata['lo']
        self.builder_data[key][sample]['data']['hi'] += moddata['hi']
        self.builder_data[key][sample]['data']['mask'] += moddata['mask']

        if thismod:
            self.required_parsets.setdefault(
                thismod['name'],
                [required_parset(defined_samp['data'], thismod['data'])],
            )

    def finalize(self):
        return self.builder_data


class normsys_combined:
    name = 'normsys'
    op_code = 'multiplication'

    def __init__(
        self, modifiers, pdfconfig, builder_data, interpcode='code1', batch_size=None
    ):
        self.interpcode = interpcode
        assert self.interpcode in ['code1', 'code4']

        keys = [f'{mtype}/{m}' for m, mtype in modifiers]
        normsys_mods = [m for m, _ in modifiers]

        self.batch_size = batch_size

        parfield_shape = (
            (self.batch_size, pdfconfig.npars)
            if self.batch_size
            else (pdfconfig.npars,)
        )
        self.param_viewer = ParamViewer(parfield_shape, pdfconfig.par_map, normsys_mods)
        self._normsys_histoset = [
            [
                [
                    builder_data[m][s]['data']['lo'],
                    builder_data[m][s]['data']['nom_data'],
                    builder_data[m][s]['data']['hi'],
                ]
                for s in pdfconfig.samples
            ]
            for m in keys
        ]
        self._normsys_mask = [
            [[builder_data[m][s]['data']['mask']] for s in pdfconfig.samples]
            for m in keys
        ]

        if normsys_mods:
            self.interpolator = getattr(interpolators, self.interpcode)(
                self._normsys_histoset
            )

        self._precompute()
        events.subscribe('tensorlib_changed')(self._precompute)

    def _precompute(self):
        if not self.param_viewer.index_selection:
            return
        tensorlib, _ = get_backend()
        self.normsys_mask = tensorlib.tile(
            tensorlib.astensor(self._normsys_mask, dtype="bool"),
            (1, 1, self.batch_size or 1, 1),
        )
        self.normsys_default = tensorlib.ones(self.normsys_mask.shape)
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
            normsys_alphaset = self.param_viewer.get(pars, self.indices)
        else:
            normsys_alphaset = self.param_viewer.get(pars)
        results_norm = self.interpolator(normsys_alphaset)

        # either rely on numerical no-op or force with line below
        results_norm = tensorlib.where(
            self.normsys_mask, results_norm, self.normsys_default
        )
        return results_norm
