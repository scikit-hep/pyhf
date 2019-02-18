import logging

from . import modifier
from ..paramsets import constrained_by_normal
from .. import get_backend, default_backend, events

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
    def __init__(self, lumi_mods, pdfconfig, mega_mods):
        self._parindices = list(range(len(pdfconfig.suggested_init())))

        pnames = [pname for pname, _ in lumi_mods]
        keys = ['{}/{}'.format(mtype, m) for m, mtype in lumi_mods]
        lumi_mods = [m for m, _ in lumi_mods]
        self._lumi_indices = [self._parindices[pdfconfig.par_slice(p)] for p in pnames]

        self._lumi_mask = [
            [[mega_mods[s][m]['data']['mask']] for s in pdfconfig.samples] for m in keys
        ]
        self._precompute()
        events.subscribe('tensorlib_changed')(self._precompute)

    def _precompute(self):
        tensorlib, _ = get_backend()
        self.lumi_mask = default_backend.astensor(self._lumi_mask)
        self.lumi_default = default_backend.ones(self.lumi_mask.shape)
        self.lumi_indices = default_backend.astensor(self._lumi_indices, dtype='int')

    def apply(self, pars):
        tensorlib, _ = get_backend()
        lumi_indices = tensorlib.astensor(self.lumi_indices, dtype='int')
        lumi_mask = tensorlib.astensor(self.lumi_mask)
        if not tensorlib.shape(lumi_indices)[0]:
            return
        lumis = tensorlib.gather(pars, lumi_indices)
        results_lumi = lumi_mask * tensorlib.reshape(
            lumis, tensorlib.shape(lumis) + (1, 1)
        )
        results_lumi = tensorlib.where(
            lumi_mask, results_lumi, tensorlib.astensor(self.lumi_default)
        )
        return results_lumi
