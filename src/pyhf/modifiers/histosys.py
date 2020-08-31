import logging

from . import modifier
from .. import get_backend, events
from .. import interpolators
from ..parameters import constrained_by_normal, ParamViewer

log = logging.getLogger(__name__)


@modifier(name='histosys', constrained=True, op_code='addition')
class histosys(object):
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


class histosys_combined(object):
    def __init__(
        self, histosys_mods, pdfconfig, mega_mods, interpcode='code0', batch_size=None
    ):
        self.batch_size = batch_size
        self.interpcode = interpcode
        assert self.interpcode in ['code0', 'code2', 'code4p']

        keys = ['{}/{}'.format(mtype, m) for m, mtype in histosys_mods]
        histosys_mods = [m for m, _ in histosys_mods]

        parfield_shape = (
            (self.batch_size, pdfconfig.npars)
            if self.batch_size
            else (pdfconfig.npars,)
        )
        self.param_viewer = ParamViewer(
            parfield_shape, pdfconfig.par_map, histosys_mods
        )

        self._histosys_histoset = [
            [
                [
                    mega_mods[m][s]['data']['lo_data'],
                    mega_mods[m][s]['data']['nom_data'],
                    mega_mods[m][s]['data']['hi_data'],
                ]
                for s in pdfconfig.samples
            ]
            for m in keys
        ]
        self._histosys_mask = [
            [[mega_mods[m][s]['data']['mask']] for s in pdfconfig.samples] for m in keys
        ]

        if histosys_mods:
            self.interpolator = getattr(interpolators, self.interpcode)(
                self._histosys_histoset
            )

        self._precompute()
        events.subscribe('tensorlib_changed')(self._precompute)

    def _precompute(self):
        if not self.param_viewer.index_selection:
            return
        tensorlib, _ = get_backend()
        self.histosys_mask = tensorlib.astensor(self._histosys_mask, dtype="bool")
        self.histosys_default = tensorlib.zeros(self.histosys_mask.shape)
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
            histosys_alphaset = self.param_viewer.get(pars, self.indices)
        else:
            histosys_alphaset = self.param_viewer.get(pars)

        results_histo = self.interpolator(histosys_alphaset)
        # either rely on numerical no-op or force with line below
        results_histo = tensorlib.where(
            self.histosys_mask, results_histo, self.histosys_default
        )
        return results_histo
