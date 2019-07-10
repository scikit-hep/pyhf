import logging

from . import modifier
from ..paramsets import constrained_by_normal
from .. import get_backend, default_backend, events
from ..utils import Parameters

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
    def __init__(self, lumi_mods, pdfconfig, mega_mods, batch_size):
        self.batch_size = batch_size

        self._parindices = list(range(len(pdfconfig.suggested_init())))


        pnames = [pname for pname, _ in lumi_mods]
        keys = ['{}/{}'.format(mtype, m) for m, mtype in lumi_mods]
        lumi_mods = [m for m, _ in lumi_mods]

        self.parameters_helper = Parameters((self.batch_size,len(pdfconfig.suggested_init()),), pdfconfig.par_map)
        self.parameters_helper._select_indices(pnames)

        self._lumi_mask = [
            [[mega_mods[s][m]['data']['mask']] for s in pdfconfig.samples] for m in keys
        ]

        self._precompute()
        events.subscribe('tensorlib_changed')(self._precompute)

    def _precompute(self):
        tensorlib, _ = get_backend()
        import numpy as np
        self.lumi_mask = default_backend.astensor(self._lumi_mask)
        self.lumi_mask = np.repeat(self.lumi_mask,self.batch_size,axis=2)
        self.lumi_default = default_backend.ones(self.lumi_mask.shape)

    def apply(self, pars):
        tensorlib, _ = get_backend()
        lumi_mask = tensorlib.astensor(self.lumi_mask)
        if not self.parameters_helper.index_selection:
            return



        lumis = tensorlib.astensor(self.parameters_helper.get_slice(pars))
        lumis = tensorlib.reshape(lumis, (1,1,self.batch_size))
        results_lumi = lumi_mask * tensorlib.reshape(
            lumis, tensorlib.shape(lumis) + (1,)
        )
        results_lumi = tensorlib.where(
            lumi_mask, results_lumi, tensorlib.astensor(self.lumi_default)
        )
        return results_lumi
