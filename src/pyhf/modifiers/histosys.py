import logging

from . import modifier
from ..paramsets import constrained_by_normal
from .. import get_backend, events
from .. import interpolators
from ..paramview import ParamViewer
log = logging.getLogger(__name__)


@modifier(name='histosys', constrained=True, op_code='addition')
class histosys(object):
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


class histosys_combined(object):
    def __init__(self, histosys_mods, pdfconfig, mega_mods, interpcode='code0'):
        self.interpcode = interpcode
        assert self.interpcode in ['code0', 'code2', 'code4p']

        pnames = [pname for pname, _ in histosys_mods]
        keys = ['{}/{}'.format(mtype, m) for m, mtype in histosys_mods]
        histosys_mods = [m for m, _ in histosys_mods]

        parfield_shape = (len(pdfconfig.suggested_init()),)
        self.parameters_helper = ParamViewer(parfield_shape, pdfconfig.par_map, pnames)

        self._histosys_histoset = [
            [
                [
                    mega_mods[s][m]['data']['lo_data'],
                    mega_mods[s][m]['data']['nom_data'],
                    mega_mods[s][m]['data']['hi_data'],
                ]
                for s in pdfconfig.samples
            ]
            for m in keys
        ]
        self._histosys_mask = [
            [[mega_mods[s][m]['data']['mask']] for s in pdfconfig.samples] for m in keys
        ]

        if histosys_mods:
            self.interpolator = getattr(interpolators, self.interpcode)(
                self._histosys_histoset
            )

        self._precompute()
        events.subscribe('tensorlib_changed')(self._precompute)

    def _precompute(self):
        tensorlib, _ = get_backend()
        self.histosys_mask = tensorlib.astensor(self._histosys_mask)
        self.histosys_default = tensorlib.zeros(self.histosys_mask.shape)

    def apply(self, pars):
        '''
        Returns:
            modification tensor: Shape (n_modifiers, n_global_samples, n_alphas, n_global_bin)
        '''
        tensorlib, _ = get_backend()
        if not self.parameters_helper.index_selection:
            return


        slices = self.parameters_helper.get_slice(pars)
        histosys_alphaset = tensorlib.astensor(
            [
                # reshape in order to go from 1 slement column
                # to flat array
                # [
                #  [a1],  -> [a1,a2]
                #  [a2],
                # ]
                tensorlib.reshape(x,(-1,)) for x in slices 
            ]
        )
        results_histo = self.interpolator(histosys_alphaset)
        # either rely on numerical no-op or force with line below
        results_histo = tensorlib.where(
            self.histosys_mask, results_histo, self.histosys_default
        )
        return results_histo
