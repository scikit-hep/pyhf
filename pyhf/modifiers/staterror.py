import logging

from . import modifier
from ..paramsets import constrained_by_normal
from .. import get_backend, default_backend, events

log = logging.getLogger(__name__)


@modifier(name='staterror', constrained=True, op_code='multiplication')
class staterror(object):
    @classmethod
    def required_parset(cls, n_parameters):
        return {
            'paramset_type': constrained_by_normal,
            'n_parameters': n_parameters,
            'modifier': cls.__name__,
            'is_constrained': cls.is_constrained,
            'is_shared': True,
            'inits': (1.0,) * n_parameters,
            'bounds': ((1e-10, 10.0),) * n_parameters,
            'auxdata': (1.0,) * n_parameters,
        }


class staterror_combined(object):
    def __init__(self, staterr_mods, pdfconfig, mega_mods):
        self._parindices = list(range(len(pdfconfig.suggested_init())))

        pnames = [pname for pname, _ in staterr_mods]
        keys = ['{}/{}'.format(mtype, m) for m, mtype in staterr_mods]
        staterr_mods = [m for m, _ in staterr_mods]

        self._staterror_indices = [
            self._parindices[pdfconfig.par_slice(p)] for p in pnames
        ]
        self._staterr_mods = staterr_mods
        self._staterror_mask = [
            [[mega_mods[s][m]['data']['mask']] for s in pdfconfig.samples] for m in keys
        ]
        self.__staterror_uncrt = default_backend.astensor(
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

        if self._staterror_indices:
            access_rows = []
            staterror_mask = default_backend.astensor(self._staterror_mask)
            for mask, inds in zip(staterror_mask, self._staterror_indices):
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
        self.staterror_mask = tensorlib.astensor(self._staterror_mask)
        self.staterror_default = tensorlib.ones(tensorlib.shape(self.staterror_mask))

        if self._staterror_indices:
            self.factor_access_indices = tensorlib.astensor(
                self._factor_access_indices, dtype='int'
            )
            self.default_value = tensorlib.astensor([1.0])
            self.sample_ones = tensorlib.ones(tensorlib.shape(self.staterror_mask)[1])
            self.alpha_ones = tensorlib.astensor([1])
        else:
            self.factor_access_indices = None

    def finalize(self, pdfconfig):
        staterror_mask = default_backend.astensor(self._staterror_mask)
        for this_mask, uncert_this_mod, mod in zip(
            staterror_mask, self.__staterror_uncrt, self._staterr_mods
        ):
            active_nominals = default_backend.where(
                this_mask[:, 0, :],
                uncert_this_mod[:, 1, :],
                default_backend.zeros(uncert_this_mod[:, 1, :].shape),
            )
            summed_nominals = default_backend.sum(active_nominals, axis=0)

            # the below tries to filter cases in which this modifier is not
            # used by checking non zeroness.. should probably use mask
            numerator = default_backend.where(
                uncert_this_mod[:, 1, :] > 0,
                uncert_this_mod[:, 0, :],
                default_backend.zeros(uncert_this_mod[:, 1, :].shape),
            )
            denominator = default_backend.where(
                summed_nominals > 0,
                summed_nominals,
                default_backend.ones(uncert_this_mod[:, 1, :].shape),
            )
            relerrs = numerator / denominator
            sigmas = default_backend.sqrt(
                default_backend.sum(default_backend.power(relerrs, 2), axis=0)
            )
            assert len(sigmas[sigmas > 0]) == pdfconfig.param_set(mod).n_parameters
            pdfconfig.param_set(mod).sigmas = default_backend.tolist(sigmas[sigmas > 0])

    def apply(self, pars):
        tensorlib, _ = get_backend()
        if self.factor_access_indices is None:
            return
        select_from = tensorlib.concatenate([pars, self.default_value])
        factor_row = tensorlib.gather(select_from, self.factor_access_indices)

        results_staterr = tensorlib.einsum(
            's,a,mb->msab',
            tensorlib.astensor(self.sample_ones),
            tensorlib.astensor(self.alpha_ones),
            factor_row,
        )

        results_staterr = tensorlib.where(
            self.staterror_mask, results_staterr, self.staterror_default
        )
        return results_staterr
