import logging

from . import modifier
from ..paramsets import constrained_by_normal
from .. import get_backend, events
from ..paramview import ParamViewer

log = logging.getLogger(__name__)


@modifier(name='lumi', constrained=True, pdf_type='normal', op_code='multiplication')
class lumi(object):
    @classmethod
    def required_parset(cls, n_parameters):
        return {
            'paramset_type': constrained_by_normal,
            'n_parameters': 1,
            'modifier': cls.__name__,
            'is_constrained': cls.is_constrained,
            'is_shared': True,
            'op_code': cls.op_code,
            'inits': None,  # lumi
            'bounds': None,  # (0, 10*lumi)
            'auxdata': None,  # lumi
            'sigmas': None,  # lumi * lumirelerror
        }


class lumi_combined(object):
    def __init__(self, lumi_mods, pdfconfig, mega_mods, batch_size=None):
        self.batch_size = batch_size

        pnames = [pname for pname, _ in lumi_mods]
        keys = ['{}/{}'.format(mtype, m) for m, mtype in lumi_mods]
        lumi_mods = [m for m, _ in lumi_mods]

        parfield_shape = (self.batch_size or 1, len(pdfconfig.suggested_init()))
        self.param_viewer = ParamViewer(parfield_shape, pdfconfig.par_map, pnames)
        self._lumi_mask = [
            [[mega_mods[s][m]['data']['mask']] for s in pdfconfig.samples] for m in keys
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
        self.lumi_default = tensorlib.ones(self.lumi_mask.shape)

    def apply(self, pars):
        '''
        Returns:
            modification tensor: Shape (n_modifiers, n_global_samples, n_alphas, n_global_bin)
        '''
        if not self.param_viewer.index_selection:
            return
        tensorlib, _ = get_backend()
        if self.batch_size is None:
            batched_pars = tensorlib.reshape(pars, (1,) + tensorlib.shape(pars))
        else:
            batched_pars = pars

        lumis = self.param_viewer.get(batched_pars)
        # lumis is [(1,batch)]

        # mask is (nsys, nsam, batch, globalbin)
        results_lumi = tensorlib.einsum('ysab,xa->ysab', self.lumi_mask, lumis)

        results_lumi = tensorlib.where(self.lumi_mask, results_lumi, self.lumi_default)
        return results_lumi
