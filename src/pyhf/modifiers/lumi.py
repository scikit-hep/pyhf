import logging

from . import modifier
from .. import get_backend, events
from ..parameters import constrained_by_normal, ParamViewer

log = logging.getLogger(__name__)


@modifier(name='lumi', constrained=True, pdf_type='normal', op_code='multiplication')
class lumi(object):
    @classmethod
    def required_parset(cls, sample_data, modifier_data):
        return {
            'paramset_type': constrained_by_normal,
            'n_parameters': 1,
            'modifier': cls.__name__,
            'is_constrained': cls.is_constrained,
            'is_shared': True,
            'op_code': cls.op_code,
            'inits': None,  # lumi
            'bounds': None,  # (0, 10*lumi)
            'fixed': False,
            'auxdata': None,  # lumi
            'sigmas': None,  # lumi * lumirelerror
        }


class lumi_combined(object):
    def __init__(self, lumi_mods, pdfconfig, mega_mods, batch_size=None):
        self.batch_size = batch_size

        keys = ['{}/{}'.format(mtype, m) for m, mtype in lumi_mods]
        lumi_mods = [m for m, _ in lumi_mods]

        parfield_shape = (
            (self.batch_size, pdfconfig.npars)
            if self.batch_size
            else (pdfconfig.npars,)
        )
        self.param_viewer = ParamViewer(parfield_shape, pdfconfig.par_map, lumi_mods)

        self._lumi_mask = [
            [[mega_mods[m][s]['data']['mask']] for s in pdfconfig.samples] for m in keys
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
