import logging

from pyhf import get_backend, events
from pyhf.parameters import ParamViewer

log = logging.getLogger(__name__)


def required_parset(sample_data, modifier_data):
    return {
        'paramset_type': 'unconstrained',
        'n_parameters': 1,
        'is_scalar': True,
        'inits': (1.0,),
        'bounds': ((0, 10),),
        'fixed': False,
    }


class normfactor_builder:
    """Builder class for collecting normfactor modifier data"""

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


class normfactor_combined:
    name = 'normfactor'
    op_code = 'multiplication'

    def __init__(self, modifiers, pdfconfig, builder_data, batch_size=None):
        self.batch_size = batch_size

        keys = [f'{mtype}/{m}' for m, mtype in modifiers]
        normfactor_mods = [m for m, _ in modifiers]

        parfield_shape = (
            (self.batch_size, pdfconfig.npars)
            if self.batch_size
            else (pdfconfig.npars,)
        )
        self.param_viewer = ParamViewer(
            parfield_shape, pdfconfig.par_map, normfactor_mods
        )

        self._normfactor_mask = [
            [[builder_data[m][s]['data']['mask']] for s in pdfconfig.samples]
            for m in keys
        ]
        self._precompute()
        events.subscribe('tensorlib_changed')(self._precompute)

    def _precompute(self):
        tensorlib, _ = get_backend()
        if not self.param_viewer.index_selection:
            return
        self.normfactor_mask = tensorlib.tile(
            tensorlib.astensor(self._normfactor_mask), (1, 1, self.batch_size or 1, 1)
        )
        self.normfactor_mask_bool = tensorlib.astensor(
            self.normfactor_mask, dtype="bool"
        )
        self.normfactor_default = tensorlib.ones(self.normfactor_mask.shape)

    def apply(self, pars):
        """
        Returns:
            modification tensor: Shape (n_modifiers, n_global_samples, n_alphas, n_global_bin)
        """
        if not self.param_viewer.index_selection:
            return
        tensorlib, _ = get_backend()
        if self.batch_size is None:
            normfactors = self.param_viewer.get(pars)
            results_normfactor = tensorlib.einsum(
                'msab,m->msab', self.normfactor_mask, normfactors
            )
        else:
            normfactors = self.param_viewer.get(pars)
            results_normfactor = tensorlib.einsum(
                'msab,ma->msab', self.normfactor_mask, normfactors
            )

        results_normfactor = tensorlib.where(
            self.normfactor_mask_bool, results_normfactor, self.normfactor_default
        )
        return results_normfactor
