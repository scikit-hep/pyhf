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
        if not self.parameters_helper.index_selection:
            return
        self.normfactor_mask = tensorlib.astensor(self._normfactor_mask)
        batch_size = 1
        self.normfactor_mask = tensorlib.tile(tensorlib.astensor(self.normfactor_mask),(1,batch_size,1,1))
        self.normfactor_default = tensorlib.ones(self.normfactor_mask.shape)

    def apply(self, pars):
        '''
        Returns:
            modification tensor: Shape (n_modifiers, n_global_samples, n_alphas, n_global_bin)
        '''
        if not self.parameters_helper.index_selection:
            return
        tensorlib, _ = get_backend()

        normfactors = tensorlib.stack(
            self.parameters_helper.get_slice(pars)
        )
        results_normfactor = self.normfactor_mask * tensorlib.reshape(
            normfactors, tensorlib.shape(normfactors) + (1, 1)
        )
        results_normfactor = tensorlib.where(
            normfactor_mask,
            results_normfactor,
            tensorlib.astensor(self.normfactor_default),
        )
        return results_normfactor
