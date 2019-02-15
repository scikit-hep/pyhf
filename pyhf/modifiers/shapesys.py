import logging

from . import modifier
from ..paramsets import constrained_by_poisson
from .. import get_backend, default_backend, events

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
    def __init__(self, shapesys_mods, pdfconfig, mega_mods):

        pnames = [pname for pname, _ in shapesys_mods]
        keys = ['{}/{}'.format(mtype, m) for m, mtype in shapesys_mods]
        shapesys_mods = [m for m, _ in shapesys_mods]

        self._shapesys_mods = shapesys_mods
        self._pnames = pnames

        self._parindices = list(range(len(pdfconfig.suggested_init())))
        self._shapesys_indices = [
            self._parindices[pdfconfig.par_slice(p)] for p in pnames
        ]
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

        if self._shapesys_indices:
            access_rows = []
            shapesys_mask = default_backend.astensor(self._shapesys_mask)
            for mask, inds in zip(shapesys_mask, self._shapesys_indices):
                summed_mask = default_backend.sum(mask[:, 0, :], axis=0)
                assert default_backend.shape(
                    summed_mask[summed_mask > 0]
                ) == default_backend.shape(default_backend.astensor(inds))
                # make masks of > 0 and == 0
                positive_mask = summed_mask > 0
                zero_mask = summed_mask == 0
                # then apply the mask
                summed_mask[positive_mask] = inds
                summed_mask[zero_mask] = len(self._parindices) - 1
                access_rows.append(summed_mask.tolist())
            self._factor_access_indices = default_backend.tolist(
                default_backend.stack(access_rows)
            )
            self.finalize(pdfconfig)
        else:
            self._factor_access_indices = None

        self._precompute()
        events.subscribe('tensorlib_changed')(self._precompute)

    def _precompute(self):
        tensorlib, _ = get_backend()
        self.shapesys_mask = tensorlib.astensor(self._shapesys_mask)
        self.shapesys_default = tensorlib.ones(tensorlib.shape(self.shapesys_mask))

        if self._shapesys_indices:
            self.factor_access_indices = tensorlib.astensor(
                self._factor_access_indices, dtype='int'
            )
            self.default_value = tensorlib.astensor([1.0])
            self.sample_ones = tensorlib.ones(tensorlib.shape(self.shapesys_mask)[1])
            self.alpha_ones = tensorlib.astensor([1])
        else:
            self.factor_access_indices = None

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
        tensorlib, _ = get_backend()
        if self.factor_access_indices is None:
            return
        tensorlib, _ = get_backend()

        factor_row = tensorlib.gather(
            tensorlib.concatenate([tensorlib.astensor(pars), self.default_value]),
            self.factor_access_indices,
        )

        results_shapesys = tensorlib.einsum(
            's,a,mb->msab',
            tensorlib.astensor(self.sample_ones),
            tensorlib.astensor(self.alpha_ones),
            factor_row,
        )

        results_shapesys = tensorlib.where(
            self.shapesys_mask, results_shapesys, self.shapesys_default
        )
        return results_shapesys
