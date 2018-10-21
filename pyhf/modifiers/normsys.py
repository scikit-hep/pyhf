import logging
log = logging.getLogger(__name__)

from . import modifier
from ..paramsets import constrained_by_normal
from .. import get_backend, events
from ..interpolate import _hfinterpolator_code1

@modifier(name='normsys', constrained=True, op_code = 'multiplication')
class normsys(object):
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

class normsys_combined(object):
    def __init__(self, normsys_mods, pdfconfig, mega_mods):
        self._parindices = list(range(len(pdfconfig.suggested_init())))
        self._normsys_indices = [self._parindices[pdfconfig.par_slice(m)] for m in normsys_mods]
        self._normsys_histoset = [
            [
                [
                    mega_mods[s][m]['data']['lo'],
                    mega_mods[s][m]['data']['nom_data'],
                    mega_mods[s][m]['data']['hi'],
                ]
                for s in pdfconfig.samples
            ] for m in normsys_mods
        ]
        self._normsys_mask = [
            [
                [
                    mega_mods[s][m]['data']['mask'],
                ]
                for s in pdfconfig.samples
            ] for m in normsys_mods
        ]

        if len(normsys_mods):
            self.interpolator = _hfinterpolator_code1(self._normsys_histoset)

        self._precompute()
        events.subscribe('tensorlib_changed')(self._precompute)

    def _precompute(self):
        tensorlib, _ = get_backend()
        self.normsys_mask = tensorlib.astensor(self._normsys_mask)
        self.normsys_default = tensorlib.ones(self.normsys_mask.shape)
        self.normsys_indices = tensorlib.astensor(self._normsys_indices, dtype='int')

    def apply(self,pars):
        tensorlib, _ = get_backend()
        if not tensorlib.shape(self.normsys_indices)[0]:
            return
        normsys_alphaset = tensorlib.gather(pars,self.normsys_indices)
        results_norm   = self.interpolator(normsys_alphaset)

        #either rely on numerical no-op or force with line below
        results_norm   = tensorlib.where(self.normsys_mask,results_norm,self.normsys_default)
        return results_norm
