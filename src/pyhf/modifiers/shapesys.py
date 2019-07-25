import logging

from . import modifier
from ..paramsets import constrained_by_poisson
from .. import get_backend, default_backend, events
from ..paramview import ParamViewer

log = logging.getLogger(__name__)


@modifier(
    name='shapesys', constrained=True, pdf_type='poisson', op_code='multiplication'
)
class shapesys(object):
    @classmethod
    def required_parset(cls, n_parameters):
        return {
            'paramset_type': constrained_by_poisson,
            'n_parameters': n_parameters,
            'modifier': cls.__name__,
            'is_constrained': cls.is_constrained,
            'is_shared': False,
            'inits': (1.0,) * n_parameters,
            'bounds': ((1e-10, 10.0),) * n_parameters,
            # nb: auxdata/factors set by finalize. Set to non-numeric to crash
            # if we fail to set auxdata/factors correctly
            'auxdata': (None,) * n_parameters,
            'factors': (None,) * n_parameters,
        }


class shapesys_combined(object):
    def __init__(self, shapesys_mods, pdfconfig, mega_mods, batch_size=1):
        self.batch_size = batch_size

        pnames = [pname for pname, _ in shapesys_mods]
        keys = ['{}/{}'.format(mtype, m) for m, mtype in shapesys_mods]
        shapesys_mods = [m for m, _ in shapesys_mods]

        self._shapesys_mods = shapesys_mods
        self._pnames = pnames

        self._parindices = list(range(len(pdfconfig.suggested_init())))
        self._shapesys_indices = [
            self._parindices[pdfconfig.par_slice(p)] for p in pnames
        ]

        parfield_shape = (self.batch_size, len(pdfconfig.suggested_init()))
        self.parameters_helper = ParamViewer(
            parfield_shape, pdfconfig.par_map, pnames, regular=False
        )

        self._shapesys_mask = [
            [[mega_mods[s][m]['data']['mask']] for s in pdfconfig.samples] for m in keys
        ]
        self.__shapesys_uncrt = default_backend.astensor(
            [
                [
                    [
                        mega_mods[s][m]['data']['uncrt'],
                        mega_mods[s][m]['data']['nom_data'],
                    ]
                    for s in pdfconfig.samples
                ]
                for m in keys
            ]
        )

        self.finalize(pdfconfig)

        global_concatenated_bin_indices = [
            [[j for c in pdfconfig.channels for j in range(pdfconfig.channel_nbins[c])]]
        ]

        self._access_field = default_backend.tile(
            global_concatenated_bin_indices, (len(pnames), self.batch_size, 1)
        )
        # access field is shape (sys, batch, globalbin)
        for s, syst_access in enumerate(self._access_field):
            for t, batch_access in enumerate(syst_access):
                selection = self.parameters_helper.index_selection[s][t]
                for b, bin_access in enumerate(batch_access):
                    self._access_field[s, t, b] = (
                        selection[bin_access] if bin_access < len(selection) else 0
                    )

        self._precompute()
        events.subscribe('tensorlib_changed')(self._precompute)

    def _precompute(self):
        tensorlib, _ = get_backend()
        if not self.parameters_helper.index_selection:
            return
        self.shapesys_mask = tensorlib.astensor(self._shapesys_mask)
        self.shapesys_mask = tensorlib.tile(
            self.shapesys_mask, (1, 1, self.batch_size, 1)
        )
        self.access_field = tensorlib.astensor(self._access_field, dtype='int')
        self.sample_ones = tensorlib.ones(tensorlib.shape(self.shapesys_mask)[1])
        self.shapesys_default = tensorlib.ones(tensorlib.shape(self.shapesys_mask))

    def finalize(self, pdfconfig):
        for uncert_this_mod, pname in zip(self.__shapesys_uncrt, self._pnames):
            unc_nom = default_backend.astensor(
                [x for x in uncert_this_mod[:, :, :] if any(x[0][x[0] > 0])]
            )
            unc = unc_nom[0, 0]
            nom = unc_nom[0, 1]
            unc_sq = default_backend.power(unc, 2)
            nom_sq = default_backend.power(nom, 2)

            # the below tries to filter cases in which
            # this modifier is not used by checking non
            # zeroness.. shoudl probably use mask
            numerator = default_backend.where(
                unc_sq > 0, nom_sq, default_backend.zeros(unc_sq.shape)
            )
            denominator = default_backend.where(
                unc_sq > 0, unc_sq, default_backend.ones(unc_sq.shape)
            )

            factors = numerator / denominator
            factors = factors[factors > 0]
            assert len(factors) == pdfconfig.param_set(pname).n_parameters
            pdfconfig.param_set(pname).factors = default_backend.tolist(factors)
            pdfconfig.param_set(pname).auxdata = default_backend.tolist(factors)

    def apply(self, pars):
        '''
        Returns:
            modification tensor: Shape (n_modifiers, n_global_samples, n_alphas, n_global_bin)
        '''
        tensorlib, _ = get_backend()
        if not self.parameters_helper.index_selection:
            return
        tensorlib, _ = get_backend()

        tensorlib, _ = get_backend()
        pars = tensorlib.astensor(pars)
        if self.batch_size == 1:
            batched_pars = tensorlib.reshape(
                pars, (self.batch_size,) + tensorlib.shape(pars)
            )
        else:
            batched_pars = pars

        flat_pars = tensorlib.reshape(batched_pars, (-1,))
        shapefactors = tensorlib.gather(flat_pars, self.access_field)
        results_shapesys = tensorlib.einsum(
            'yab,s->ysab', shapefactors, self.sample_ones
        )

        results_shapesys = tensorlib.where(
            self.shapesys_mask, results_shapesys, self.shapesys_default
        )
        return results_shapesys
