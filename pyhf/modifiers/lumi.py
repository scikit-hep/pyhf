import logging

from . import modifier
from ..paramsets import constrained_by_normal
from .. import get_backend, default_backend, events

log = logging.getLogger(__name__)


@modifier(
    name='lumi', constrained=True, pdf_type='normal', op_code='multiplication'
)
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
            'inits': (0.0,),
            'bounds': ((-5.0, 5.0),),
            'auxdata': (0.0,),
        }


class lumi_combined(object):
    def __init__(self, lumi_mods, pdfconfig, mega_mods):
        self._lumi_mods = lumi_mods
        self._parindices = list(range(len(pdfconfig.suggested_init())))
        self._lumi_indices = [
            self._parindices[pdfconfig.par_slice(m)] for m in lumi_mods
        ]
        self._lumi_mask = [
            [[mega_mods[s][m]['data']['mask']] for s in pdfconfig.samples]
            for m in lumi_mods
        ]

        self._precompute()
        events.subscribe('tensorlib_changed')(self._precompute)

    def _precompute(self):
        tensorlib, _ = get_backend()
        self.lumi_mask = tensorlib.astensor(self._lumi_mask)
        self.lumi_default = tensorlib.ones(tensorlib.shape(self.lumi_mask))

        self.default_value = tensorlib.astensor([1.0])
        self.sample_ones = tensorlib.ones(tensorlib.shape(self.lumi_mask)[1])
        self.alpha_ones = tensorlib.astensor([1])

    def apply(self, pars):
        tensorlib, _ = get_backend()

        if not tensorlib.shape(self.lumi_indices)[0]:
            return

        return
