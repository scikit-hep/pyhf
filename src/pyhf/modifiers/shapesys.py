import logging

from . import modifier
from .. import get_backend, default_backend, events
from ..parameters import constrained_by_poisson, ParamViewer

log = logging.getLogger(__name__)


@modifier(
    name='shapesys', constrained=True, pdf_type='poisson', op_code='multiplication'
)
class shapesys(object):
    @classmethod
    def required_parset(cls, sample_data, modifier_data):
        # count the number of bins with nonzero, positive yields
        valid_bins = [
            (sample_bin > 0 and modifier_bin > 0)
            for sample_bin, modifier_bin in zip(modifier_data, sample_data)
        ]
        n_parameters = sum(valid_bins)
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
    def __init__(self, shapesys_mods, pdfconfig, mega_mods, batch_size=None):
        self.batch_size = batch_size

        keys = ['{}/{}'.format(mtype, m) for m, mtype in shapesys_mods]
        self._shapesys_mods = [m for m, _ in shapesys_mods]

        parfield_shape = (self.batch_size or 1, pdfconfig.npars)
        self.param_viewer = ParamViewer(
            parfield_shape, pdfconfig.par_map, self._shapesys_mods
        )

        self._shapesys_mask = [
            [[mega_mods[m][s]['data']['mask']] for s in pdfconfig.samples] for m in keys
        ]
        self.__shapesys_info = default_backend.astensor(
            [
                [
                    [
                        mega_mods[m][s]['data']['mask'],
                        mega_mods[m][s]['data']['nom_data'],
                        mega_mods[m][s]['data']['uncrt'],
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
            global_concatenated_bin_indices,
            (len(shapesys_mods), self.batch_size or 1, 1),
        )
        # access field is shape (sys, batch, globalbin)

        # reindex it based on current masking
        self._reindex_access_field(pdfconfig)

        self._precompute()
        events.subscribe('tensorlib_changed')(self._precompute)

    def _reindex_access_field(self, pdfconfig):
        for syst_index, syst_access in enumerate(self._access_field):
            if not pdfconfig.param_set(self._shapesys_mods[syst_index]).n_parameters:
                self._access_field[syst_index] = 0
                continue
            for batch_index, batch_access in enumerate(syst_access):
                selection = self.param_viewer.index_selection[syst_index][batch_index]
                access_field_for_syst_and_batch = default_backend.zeros(
                    len(batch_access)
                )
                singular_sample_index = [
                    idx
                    for idx, syst in enumerate(
                        default_backend.astensor(self._shapesys_mask)[syst_index, :, 0]
                    )
                    if any(syst)
                ][-1]
                sample_mask = self._shapesys_mask[syst_index][singular_sample_index][0]
                access_field_for_syst_and_batch[sample_mask] = selection
                self._access_field[
                    syst_index, batch_index
                ] = access_field_for_syst_and_batch

    def _precompute(self):
        tensorlib, _ = get_backend()
        if not self.param_viewer.index_selection:
            return
        self.shapesys_mask = tensorlib.astensor(self._shapesys_mask, dtype="bool")
        self.shapesys_mask = tensorlib.tile(
            self.shapesys_mask, (1, 1, self.batch_size or 1, 1)
        )
        self.access_field = tensorlib.astensor(self._access_field, dtype='int')
        self.sample_ones = tensorlib.ones(tensorlib.shape(self.shapesys_mask)[1])
        self.shapesys_default = tensorlib.ones(tensorlib.shape(self.shapesys_mask))

    def finalize(self, pdfconfig):
        # self.__shapesys_info: (parameter, sample, (mask, nominal rate, uncertainty), bin)
        for mod_uncert_info, pname in zip(self.__shapesys_info, self._shapesys_mods):
            if not pdfconfig.param_set(pname).n_parameters:
                continue

            # identify the sample that the given parameter affects
            # shapesys is not shared, so there should only ever be at most one sample
            sample_uncert_info = mod_uncert_info[
                default_backend.sum(mod_uncert_info[:, 0] > 0, axis=1, dtype='bool')
            ]

            # if empty: a case where given shapesys modifier affects zero samples
            if default_backend.shape(sample_uncert_info)[0] == 0:
                continue

            # sample_uncert_info = (bin_mask, nominal rate, uncertainty)
            bin_mask = default_backend.astensor(sample_uncert_info[0][0], dtype='bool')
            nom_unc = sample_uncert_info[0, 1:]

            # Why this works: setting the default/not-affecting-sample == -1
            # means that squaring it gives +1 by default)

            # compute gamma**2 and sigma**2
            nom_unc_sq = default_backend.power(nom_unc, 2)
            # when the nominal rate = 0 OR uncertainty = 0, set = 1
            nom_unc_sq[nom_unc_sq == 0] = 1
            # gamma**2 / sigma**2
            factors = (nom_unc_sq[0] / nom_unc_sq[1])[bin_mask]
            assert len(factors) == pdfconfig.param_set(pname).n_parameters

            pdfconfig.param_set(pname).factors = default_backend.tolist(factors)
            pdfconfig.param_set(pname).auxdata = default_backend.tolist(factors)

    def apply(self, pars):
        '''
        Returns:
            modification tensor: Shape (n_modifiers, n_global_samples, n_alphas, n_global_bin)
        '''
        tensorlib, _ = get_backend()
        if not self.param_viewer.index_selection:
            return
        tensorlib, _ = get_backend()
        if self.batch_size is None:
            flat_pars = pars
        else:
            flat_pars = tensorlib.reshape(pars, (-1,))
        shapefactors = tensorlib.gather(flat_pars, self.access_field)
        results_shapesys = tensorlib.einsum(
            'mab,s->msab', shapefactors, self.sample_ones
        )

        results_shapesys = tensorlib.where(
            self.shapesys_mask, results_shapesys, self.shapesys_default
        )
        return results_shapesys
