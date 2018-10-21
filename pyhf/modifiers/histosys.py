import logging
log = logging.getLogger(__name__)

from . import modifier
from ..paramsets import constrained_by_normal
from .. import get_backend, events
from ..interpolate import _hfinterpolator_code0

@modifier(name='histosys', constrained=True, op_code = 'addition')
class histosys(object):
    @classmethod
    def required_parset(cls, n_parameters):
        n_parameters = 1
        return {
            'constraint': constrained_by_normal,
            'n_parameters': n_parameters,
            'modifier': cls.__name__,
            'is_constrained': cls.is_constrained,
            'is_shared': True,
            'op_code': cls.op_code,
            'inits': (0.0,) * n_parameters,
            'bounds': ((-5., 5.),) * n_parameters,
            'auxdata': (0.,) * n_parameters,
            'factors': () * n_parameters
        }

class histosys_combined(object):
    def __init__(self,histosys_mods,pdfconfig,mega_mods):
        self._parindices = list(range(len(pdfconfig.suggested_init())))
        self._histo_indices = [self._parindices[pdfconfig.par_slice(m)] for m in histosys_mods]
        self._histosys_histoset = [
            [
                [
                    mega_mods[s][m]['data']['lo_data'],
                    mega_mods[s][m]['data']['nom_data'],
                    mega_mods[s][m]['data']['hi_data'],
                ]
                for s in pdfconfig.samples
            ] for m in histosys_mods
        ]
        self._histosys_mask = [
            [
                [
                    mega_mods[s][m]['data']['mask'],
                ]
                for s in pdfconfig.samples
            ] for m in histosys_mods
        ]

        if len(histosys_mods):
            self.interpolator = _hfinterpolator_code0(self._histosys_histoset)

        self._precompute()
        events.subscribe('tensorlib_changed')(self._precompute)

    def _precompute(self):
        tensorlib, _ = get_backend()
        self.histosys_mask = tensorlib.astensor(self._histosys_mask)
        self.histosys_default = tensorlib.zeros(self.histosys_mask.shape)
        self.histo_indices = tensorlib.astensor(self._histo_indices, dtype='int')

    def apply(self,pars):
        tensorlib, _ = get_backend()
        if not tensorlib.shape(self.histo_indices)[0]:
            return
        histosys_alphaset = tensorlib.gather(pars,self.histo_indices)
        results_histo   = self.interpolator(histosys_alphaset)
        # either rely on numerical no-op or force with line below
        results_histo   = tensorlib.where(self.histosys_mask,results_histo,self.histosys_default)
        return results_histo
