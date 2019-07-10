import logging

from . import modifier
from ..paramsets import unconstrained
from .. import get_backend, default_backend, events
from ..utils import Parameters
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
    def __init__(self, normfactor_mods, pdfconfig, mega_mods, batch_size):
        self.batch_size = batch_size
        self._parindices = list(range(len(pdfconfig.suggested_init())))

        pnames = [pname for pname, _ in normfactor_mods]
        keys = ['{}/{}'.format(mtype, m) for m, mtype in normfactor_mods]
        normfactor_mods = [m for m, _ in normfactor_mods]

        self.parameters_helper = Parameters((self.batch_size,len(pdfconfig.suggested_init()),), pdfconfig.par_map)
        self.parameters_helper._select_indices(pnames)
        self._normfactor_mask = [
            [[mega_mods[s][m]['data']['mask']] for s in pdfconfig.samples] for m in keys
        ]
        self._precompute()
        events.subscribe('tensorlib_changed')(self._precompute)

    def _precompute(self):
        tensorlib, _ = get_backend()
        import numpy as np
        self.normfactor_mask = default_backend.astensor(self._normfactor_mask)
        self.normfactor_mask = np.repeat(self.normfactor_mask,self.batch_size,axis=2)
        self.normfactor_default = default_backend.ones(self.normfactor_mask.shape)

    def apply(self, pars):
        tensorlib, _ = get_backend()
        normfactor_mask = tensorlib.astensor(self.normfactor_mask)
        if not self.parameters_helper.index_selection:
            return
        what = self.parameters_helper.get_slice(pars)
        what = tensorlib.reshape(what, (tensorlib.shape(normfactor_mask)[0],1,self.batch_size))
        normfactors = tensorlib.astensor(what)
        #TODO: explain why astensor here assumes some regularity about the slices   z
        results_normfactor = normfactor_mask * tensorlib.reshape(
            normfactors, tensorlib.shape(normfactors) + (1,)
        )
        results_normfactor = tensorlib.where(
            normfactor_mask,
            results_normfactor,
            tensorlib.astensor(self.normfactor_default),
        )
        return results_normfactor
