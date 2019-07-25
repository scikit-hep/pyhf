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
    def __init__(self, lumi_mods, pdfconfig, mega_mods, batch_size=1):
        self.batch_size = batch_size

        pnames = [pname for pname, _ in lumi_mods]
        keys = ['{}/{}'.format(mtype, m) for m, mtype in lumi_mods]
        lumi_mods = [m for m, _ in lumi_mods]

        parfield_shape = (self.batch_size, len(pdfconfig.suggested_init()))
        self.parameters_helper = ParamViewer(parfield_shape, pdfconfig.par_map, pnames)
        self._lumi_mask = [
            [[mega_mods[s][m]['data']['mask']] for s in pdfconfig.samples] for m in keys
        ]
        self._precompute()
        events.subscribe('tensorlib_changed')(self._precompute)

    def _precompute(self):
        if not self.parameters_helper.index_selection:
            return
        tensorlib, _ = get_backend()
        self.lumi_mask = tensorlib.tile(
            tensorlib.astensor(self._lumi_mask), (1, 1, self.batch_size, 1)
        )
        self.lumi_default = tensorlib.ones(self.lumi_mask.shape)

    def apply(self, pars):
        '''
        Returns:
            modification tensor: Shape (n_modifiers, n_global_samples, n_alphas, n_global_bin)
        '''
        if not self.parameters_helper.index_selection:
            return
        tensorlib, _ = get_backend()
        if self.batch_size == 1:
            batched_pars = tensorlib.reshape(
                pars, (self.batch_size,) + tensorlib.shape(pars)
            )
        else:
            batched_pars = pars

        lumi_mask = tensorlib.astensor(self.lumi_mask)

        lumis = self.parameters_helper.get_slice(batched_pars)[0]
        # lumis is [(batch, 1)]

        # mask is (nsys, nsam, batch, globalbin)
        results_lumi = tensorlib.einsum('ysab,ax->ysab', lumi_mask, lumis)

        results_lumi = tensorlib.where(
            lumi_mask, results_lumi, tensorlib.astensor(self.lumi_default)
        )
        return results_lumi
