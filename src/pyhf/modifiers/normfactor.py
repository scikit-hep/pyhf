import logging

from . import modifier
from ..paramsets import unconstrained
from .. import get_backend, default_backend, events

log = logging.getLogger(__name__)


@modifier(name='normfactor', op_code='multiplication')
class normfactor(object):
    @classmethod
    def required_parset(cls, n_parameters):
        return {
            'paramset_type': unconstrained,
            'n_parameters': 1,
            'modifier': cls.__name__,
            'is_constrained': cls.is_constrained,
            'is_shared': True,
            'inits': (1.0,),
            'bounds': ((0, 10),),
        }


class normfactor_combined(object):
    def __init__(self, normfactor_mods, pdfconfig, mega_mods):
        self._parindices = list(range(len(pdfconfig.suggested_init())))

        pnames = [pname for pname, _ in normfactor_mods]
        keys = ['{}/{}'.format(mtype, m) for m, mtype in normfactor_mods]
        normfactor_mods = [m for m, _ in normfactor_mods]

        self._normfactor_indices = [
            self._parindices[pdfconfig.par_slice(p)] for p in pnames
        ]
        self._normfactor_mask = [
            [[mega_mods[s][m]['data']['mask']] for s in pdfconfig.samples] for m in keys
        ]
        self._precompute()
        events.subscribe('tensorlib_changed')(self._precompute)

    def _precompute(self):
        tensorlib, _ = get_backend()
        self.normfactor_mask = default_backend.astensor(self._normfactor_mask)
        self.normfactor_default = default_backend.ones(self.normfactor_mask.shape)
        self.normfactor_indices = default_backend.astensor(
            self._normfactor_indices, dtype='int'
        )

    def apply(self, pars):
        tensorlib, _ = get_backend()
        normfactor_indices = tensorlib.astensor(self.normfactor_indices, dtype='int')
        normfactor_mask = tensorlib.astensor(self.normfactor_mask)
        if not tensorlib.shape(normfactor_indices)[0]:
            return
        normfactors = tensorlib.gather(pars, normfactor_indices)
        results_normfactor = normfactor_mask * tensorlib.reshape(
            normfactors, tensorlib.shape(normfactors) + (1, 1)
        )
        results_normfactor = tensorlib.where(
            normfactor_mask,
            results_normfactor,
            tensorlib.astensor(self.normfactor_default),
        )
        return results_normfactor
