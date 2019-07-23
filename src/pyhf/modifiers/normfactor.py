import logging

from . import modifier
from ..paramsets import unconstrained
from .. import get_backend, default_backend, events
from ..paramview import ParamViewer

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
        pnames = [pname for pname, _ in normfactor_mods]
        keys = ['{}/{}'.format(mtype, m) for m, mtype in normfactor_mods]
        normfactor_mods = [m for m, _ in normfactor_mods]

        parfield_shape = (len(pdfconfig.suggested_init()),)
        self.parameters_helper = ParamViewer(parfield_shape, pdfconfig.par_map, pnames)
        self._normfactor_mask = [
            [[mega_mods[s][m]['data']['mask']] for s in pdfconfig.samples] for m in keys
        ]
        self._precompute()
        events.subscribe('tensorlib_changed')(self._precompute)

    def _precompute(self):
        tensorlib, _ = get_backend()
        batch_size = 1
        self.normfactor_mask = tensorlib.tile(tensorlib.astensor(self._normfactor_mask),(1,batch_size,1,1))
        self.normfactor_default = tensorlib.ones(tensorlib.shape(self.normfactor_mask))

    def apply(self, pars):
        '''
        Returns:
            modification tensor: Shape (n_modifiers, n_global_samples, n_alphas, n_global_bin)
        '''
        tensorlib, _ = get_backend()
        normfactor_mask = tensorlib.astensor(self.normfactor_mask)
        if not self.parameters_helper.index_selection:
            return
        normfactors = self.parameters_helper.get_slice(pars) # list of (slicesize,) (if batched it's (batch, slicesize))
        
        normfactors = tensorlib.astensor(normfactors)
        shaped = tensorlib.reshape(
            normfactors, (tensorlib.shape(normfactors)[0],) + (1,1,1)    #last 2 dim (alphasets,  bins) set to broadcast)
        )

        results_normfactor = normfactor_mask * shaped
        results_normfactor = tensorlib.where(
            normfactor_mask,
            results_normfactor,
            tensorlib.astensor(self.normfactor_default),
        )
        return results_normfactor
