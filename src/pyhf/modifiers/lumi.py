import logging

from pyhf import get_backend, events
from pyhf.parameters import ParamViewer

log = logging.getLogger(__name__)


def required_parset(sample_data, modifier_data):
    return {
        'paramset_type': 'constrained_by_normal',
        'n_parameters': 1,
        'is_scalar': True,
        'inits': None,  # lumi
        'bounds': None,  # (0, 10*lumi)
        'fixed': False,
        'auxdata': None,  # lumi
        'sigmas': None,  # lumi * lumirelerror
    }


class lumi_builder:
    """Builder class for collecting lumi modifier data"""

    is_shared = True

    def __init__(self, config):
        self.builder_data = {}
        self.config = config
        self.required_parsets = {}

    def collect(self, thismod, nom):
        maskval = True if thismod else False
        mask = [maskval] * len(nom)
        return {'mask': mask}

    def append(self, key, channel, sample, thismod, defined_samp):
        self.builder_data.setdefault(key, {}).setdefault(sample, {}).setdefault(
            'data', {'mask': []}
        )
        nom = (
            defined_samp['data']
            if defined_samp
            else [0.0] * self.config.channel_nbins[channel]
        )
        moddata = self.collect(thismod, nom)
        self.builder_data[key][sample]['data']['mask'] += moddata['mask']
        if thismod:
            self.required_parsets.setdefault(
                thismod['name'],
                [required_parset(defined_samp['data'], thismod['data'])],
            )

    def finalize(self):
        return self.builder_data


class lumi_combined:
    name = 'lumi'
    op_code = 'multiplication'

    def __init__(self, modifiers, pdfconfig, builder_data, batch_size=None):
        self.batch_size = batch_size

        keys = [f'{mtype}/{m}' for m, mtype in modifiers]
        lumi_mods = [m for m, _ in modifiers]

        parfield_shape = (
            (self.batch_size, pdfconfig.npars)
            if self.batch_size
            else (pdfconfig.npars,)
        )
        self.param_viewer = ParamViewer(parfield_shape, pdfconfig.par_map, lumi_mods)

        self._lumi_mask = [
            [[builder_data[m][s]['data']['mask']] for s in pdfconfig.samples]
            for m in keys
        ]
        self._precompute()
        events.subscribe('tensorlib_changed')(self._precompute)

    def _precompute(self):
        if not self.param_viewer.index_selection:
            return
        tensorlib, _ = get_backend()
        self.lumi_mask = tensorlib.tile(
            tensorlib.astensor(self._lumi_mask), (1, 1, self.batch_size or 1, 1)
        )
        self.lumi_mask_bool = tensorlib.astensor(self.lumi_mask, dtype="bool")
        self.lumi_default = tensorlib.ones(self.lumi_mask.shape)

    def apply(self, pars):
        """
        Returns:
            modification tensor: Shape (n_modifiers, n_global_samples, n_alphas, n_global_bin)
        """
        if not self.param_viewer.index_selection:
            return

        tensorlib, _ = get_backend()
        lumis = self.param_viewer.get(pars)
        if self.batch_size is None:
            results_lumi = tensorlib.einsum('msab,x->msab', self.lumi_mask, lumis)
        else:
            results_lumi = tensorlib.einsum('msab,xa->msab', self.lumi_mask, lumis)

        return tensorlib.where(self.lumi_mask_bool, results_lumi, self.lumi_default)
