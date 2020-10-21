import logging

from . import modifier
from .. import get_backend, events
from ..parameters import unconstrained, ParamViewer

log = logging.getLogger(__name__)


@modifier(name='normfactor', op_code='multiplication')
class normfactor(object):
    @classmethod
    def required_parset(cls, sample_data, modifier_data):
        return {
            'paramset_type': unconstrained,
            'n_parameters': 1,
            'modifier': cls.__name__,
            'is_constrained': cls.is_constrained,
            'is_shared': True,
            'inits': (1.0,),
            'bounds': ((0, 10),),
            'fixed': False,
        }


class normfactor_combined(object):
    def __init__(self, normfactor_mods, pdfconfig, mega_mods, batch_size=None):
        self.batch_size = batch_size

        keys = ['{}/{}'.format(mtype, m) for m, mtype in normfactor_mods]
        normfactor_mods = [m for m, _ in normfactor_mods]

        parfield_shape = (
            (self.batch_size, pdfconfig.npars)
            if self.batch_size
            else (pdfconfig.npars,)
        )
        self.param_viewer = ParamViewer(
            parfield_shape, pdfconfig.par_map, normfactor_mods
        )

        self._normfactor_mask = [
            [[mega_mods[m][s]['data']['mask']] for s in pdfconfig.samples] for m in keys
        ]
        self._precompute()
        events.subscribe('tensorlib_changed')(self._precompute)

    def _precompute(self):
        tensorlib, _ = get_backend()
        if not self.param_viewer.index_selection:
            return
        self.normfactor_mask = tensorlib.tile(
            tensorlib.astensor(self._normfactor_mask), (1, 1, self.batch_size or 1, 1)
        )
        self.normfactor_mask_bool = tensorlib.astensor(
            self.normfactor_mask, dtype="bool"
        )
        self.normfactor_default = tensorlib.ones(self.normfactor_mask.shape)

    def apply(self, pars):
        """
        Returns:
            modification tensor: Shape (n_modifiers, n_global_samples, n_alphas, n_global_bin)
        """
        if not self.param_viewer.index_selection:
            return
        tensorlib, _ = get_backend()
        if self.batch_size is None:
            normfactors = self.param_viewer.get(pars)
            results_normfactor = tensorlib.einsum(
                'msab,m->msab', self.normfactor_mask, normfactors
            )
        else:
            normfactors = self.param_viewer.get(pars)
            results_normfactor = tensorlib.einsum(
                'msab,ma->msab', self.normfactor_mask, normfactors
            )

        results_normfactor = tensorlib.where(
            self.normfactor_mask_bool, results_normfactor, self.normfactor_default
        )
        return results_normfactor
