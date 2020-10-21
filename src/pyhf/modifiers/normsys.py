import logging

from . import modifier
from .. import get_backend, events
from .. import interpolators
from ..parameters import constrained_by_normal, ParamViewer

log = logging.getLogger(__name__)


@modifier(name='normsys', constrained=True, op_code='multiplication')
class normsys(object):
    @classmethod
    def required_parset(cls, sample_data, modifier_data):
        return {
            'paramset_type': constrained_by_normal,
            'n_parameters': 1,
            'modifier': cls.__name__,
            'is_constrained': cls.is_constrained,
            'is_shared': True,
            'inits': (0.0,),
            'bounds': ((-5.0, 5.0),),
            'fixed': False,
            'auxdata': (0.0,),
        }


class normsys_combined(object):
    def __init__(
        self, normsys_mods, pdfconfig, mega_mods, interpcode='code1', batch_size=None
    ):
        self.interpcode = interpcode
        assert self.interpcode in ['code1', 'code4']

        keys = ['{}/{}'.format(mtype, m) for m, mtype in normsys_mods]
        normsys_mods = [m for m, _ in normsys_mods]

        self.batch_size = batch_size

        parfield_shape = (
            (self.batch_size, pdfconfig.npars)
            if self.batch_size
            else (pdfconfig.npars,)
        )
        self.param_viewer = ParamViewer(parfield_shape, pdfconfig.par_map, normsys_mods)
        self._normsys_histoset = [
            [
                [
                    mega_mods[m][s]['data']['lo'],
                    mega_mods[m][s]['data']['nom_data'],
                    mega_mods[m][s]['data']['hi'],
                ]
                for s in pdfconfig.samples
            ]
            for m in keys
        ]
        self._normsys_mask = [
            [[mega_mods[m][s]['data']['mask']] for s in pdfconfig.samples] for m in keys
        ]

        if normsys_mods:
            self.interpolator = getattr(interpolators, self.interpcode)(
                self._normsys_histoset
            )

        self._precompute()
        events.subscribe('tensorlib_changed')(self._precompute)

    def _precompute(self):
        if not self.param_viewer.index_selection:
            return
        tensorlib, _ = get_backend()
        self.normsys_mask = tensorlib.tile(
            tensorlib.astensor(self._normsys_mask, dtype="bool"),
            (1, 1, self.batch_size or 1, 1),
        )
        self.normsys_default = tensorlib.ones(self.normsys_mask.shape)
        if self.batch_size is None:
            self.indices = tensorlib.reshape(
                self.param_viewer.indices_concatenated, (-1, 1)
            )

    def apply(self, pars):
        """
        Returns:
            modification tensor: Shape (n_modifiers, n_global_samples, n_alphas, n_global_bin)
        """
        if not self.param_viewer.index_selection:
            return

        tensorlib, _ = get_backend()
        if self.batch_size is None:
            normsys_alphaset = self.param_viewer.get(pars, self.indices)
        else:
            normsys_alphaset = self.param_viewer.get(pars)
        results_norm = self.interpolator(normsys_alphaset)

        # either rely on numerical no-op or force with line below
        results_norm = tensorlib.where(
            self.normsys_mask, results_norm, self.normsys_default
        )
        return results_norm
