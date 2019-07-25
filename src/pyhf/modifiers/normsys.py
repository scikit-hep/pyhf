import logging

from . import modifier
from ..paramsets import constrained_by_normal
from .. import get_backend, events
from .. import interpolators
from ..paramview import ParamViewer

log = logging.getLogger(__name__)


@modifier(name='normsys', constrained=True, op_code='multiplication')
class normsys(object):
    @classmethod
    def required_parset(cls, n_parameters):
        return {
            'paramset_type': constrained_by_normal,
            'n_parameters': 1,
            'modifier': cls.__name__,
            'is_constrained': cls.is_constrained,
            'is_shared': True,
            'inits': (0.0,),
            'bounds': ((-5.0, 5.0),),
            'auxdata': (0.0,),
        }


class normsys_combined(object):
    def __init__(
        self, normsys_mods, pdfconfig, mega_mods, interpcode='code1', batch_size=1
    ):
        self.interpcode = interpcode
        assert self.interpcode in ['code1', 'code4']

        pnames = [pname for pname, _ in normsys_mods]
        keys = ['{}/{}'.format(mtype, m) for m, mtype in normsys_mods]
        normsys_mods = [m for m, _ in normsys_mods]

        self.batch_size = batch_size

        parfield_shape = (self.batch_size, len(pdfconfig.suggested_init()))
        self.parameters_helper = ParamViewer(parfield_shape, pdfconfig.par_map, pnames)
        self._normsys_histoset = [
            [
                [
                    mega_mods[s][m]['data']['lo'],
                    mega_mods[s][m]['data']['nom_data'],
                    mega_mods[s][m]['data']['hi'],
                ]
                for s in pdfconfig.samples
            ]
            for m in keys
        ]
        self._normsys_mask = [
            [[mega_mods[s][m]['data']['mask']] for s in pdfconfig.samples] for m in keys
        ]

        if normsys_mods:
            self.interpolator = getattr(interpolators, self.interpcode)(
                self._normsys_histoset
            )

        self._precompute()
        events.subscribe('tensorlib_changed')(self._precompute)

    def _precompute(self):
        if not self.parameters_helper.index_selection:
            return
        tensorlib, _ = get_backend()
        self.normsys_mask = tensorlib.astensor(self._normsys_mask)
        self.normsys_mask = tensorlib.tile(
            self.normsys_mask, (1, 1, self.batch_size, 1)
        )
        self.normsys_default = tensorlib.ones(self.normsys_mask.shape)

    def apply(self, pars):
        '''
        Returns:
            modification tensor: Shape (n_modifiers, n_global_samples, n_alphas, n_global_bin)
        '''
        if not self.parameters_helper.index_selection:
            return

        tensorlib, _ = get_backend()
        pars = tensorlib.astensor(pars)
        if self.batch_size == 1:
            batched_pars = tensorlib.reshape(
                pars, (self.batch_size,) + tensorlib.shape(pars)
            )
        else:
            batched_pars = pars
        slices = self.parameters_helper.get_slice(batched_pars)

        # slices is [(batch, slicesize)] = [(batch,1)]
        normsys_alphaset = slices
        normsys_alphaset = tensorlib.reshape(
            normsys_alphaset, tensorlib.shape(normsys_alphaset)[:2]
        )

        results_norm = self.interpolator(normsys_alphaset)

        # either rely on numerical no-op or force with line below
        results_norm = tensorlib.where(
            self.normsys_mask, results_norm, self.normsys_default
        )
        return results_norm
