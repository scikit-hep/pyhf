from . import get_backend, default_backend
from . import events
from .paramview import ParamViewer


class gaussian_constraint_combined(object):
    def __init__(self, pdfconfig, batch_size=None):
        self.batch_size = batch_size
        # iterate over all constraints order doesn't matter....

        self.data_indices = list(range(len(pdfconfig.auxdata)))
        self.parsets = [pdfconfig.param_set(cname) for cname in pdfconfig.auxdata_order]

        pnames = [
            cname
            for cname in pdfconfig.auxdata_order
            if pdfconfig.param_set(cname).pdf_type == 'normal'
        ]

        parfield_shape = (self.batch_size or 1, len(pdfconfig.suggested_init()))
        self.param_viewer = ParamViewer(parfield_shape, pdfconfig.par_map, pnames)

        start_index = 0
        normal_constraint_data = []
        normal_constraint_sigmas = []
        # loop over parameters (in auxdata order) and collect
        # means / sigmas of constraint term as well as data
        # skip parsets that are not constrained by onrmal
        for parset in self.parsets:
            end_index = start_index + parset.n_parameters
            thisauxdata = self.data_indices[start_index:end_index]
            start_index = end_index
            if not parset.pdf_type == 'normal':
                continue

            normal_constraint_data.append(thisauxdata)

            # many constraints are defined on a unit gaussian
            # but we reserved the possibility that a paramset
            # can define non-standard uncertainties. This is used
            # by the paramset associated to staterror modifiers.
            # Such parsets define a 'sigmas' attribute
            try:
                normal_constraint_sigmas.append(parset.sigmas)
            except AttributeError:
                normal_constraint_sigmas.append([1.0] * len(thisauxdata))

        # if this constraint terms is at all used (non-zrto idx selection
        # start preparing constant tensors
        if self.param_viewer.index_selection:
            self._normal_data = default_backend.astensor(
                default_backend.concatenate(normal_constraint_data), dtype='int'
            )

            _normal_sigmas = default_backend.concatenate(normal_constraint_sigmas)
            sigmas = default_backend.reshape(_normal_sigmas, (1, -1))  # (1, normals)
            self._batched_sigmas = default_backend.tile(
                sigmas, (self.batch_size or 1, 1)
            )

            access_field = default_backend.concatenate(
                self.param_viewer.index_selection, axis=1
            )
            self._access_field = access_field

        else:
            self._normal_data = None
            self._batched_sigmas = None
            self._access_field = None

        self._precompute()
        events.subscribe('tensorlib_changed')(self._precompute)

    def _precompute(self):
        if not self.param_viewer.index_selection:
            return
        tensorlib, _ = get_backend()
        self.batched_sigmas = tensorlib.astensor(self._batched_sigmas)
        self.normal_data = tensorlib.astensor(self._normal_data, dtype='int')
        self.access_field = tensorlib.astensor(self._access_field, dtype='int')

    def logpdf(self, auxdata, pars):
        tensorlib, _ = get_backend()
        if not self.param_viewer.index_selection:
            return (
                tensorlib.zeros(self.batch_size)
                if self.batch_size is not None
                else tensorlib.astensor(0.0)[0]
            )

        pars = tensorlib.astensor(pars)
        if self.batch_size == 1 or self.batch_size is None:
            batched_pars = tensorlib.reshape(
                pars, (self.batch_size or 1,) + tensorlib.shape(pars)
            )
        else:
            batched_pars = pars

        flat_pars = tensorlib.reshape(batched_pars, (-1,))
        normal_means = tensorlib.gather(flat_pars, self.access_field)

        # pdf pars are done, now get data and compute
        auxdata = tensorlib.astensor(auxdata)
        normal_data = tensorlib.gather(auxdata, self.normal_data)
        normal = tensorlib.normal_logpdf(normal_data, normal_means, self.batched_sigmas)
        result = tensorlib.sum(normal, axis=1)
        if self.batch_size is None:
            return result[0]
        return result


class poisson_constraint_combined(object):
    def __init__(self, pdfconfig, batch_size=None):
        self.batch_size = batch_size
        # iterate over all constraints order doesn't matter....

        self.par_indices = list(range(len(pdfconfig.suggested_init())))
        self.data_indices = list(range(len(pdfconfig.auxdata)))
        self.parsets = [pdfconfig.param_set(cname) for cname in pdfconfig.auxdata_order]

        pnames = [
            cname
            for cname in pdfconfig.auxdata_order
            if pdfconfig.param_set(cname).pdf_type == 'poisson'
        ]

        parfield_shape = (self.batch_size or 1, len(pdfconfig.suggested_init()))
        self.param_viewer = ParamViewer(parfield_shape, pdfconfig.par_map, pnames)

        start_index = 0
        poisson_constraint_data = []
        poisson_constraint_rate_factors = []
        for parset in self.parsets:
            end_index = start_index + parset.n_parameters
            thisauxdata = self.data_indices[start_index:end_index]
            start_index = end_index
            if not parset.pdf_type == 'poisson':
                continue

            poisson_constraint_data.append(thisauxdata)

            # poisson constraints can specify a scaling factor for the
            # backgrounds rates (see: on-off problem with a aux measurement
            # with tau*b). If such a scale factor is not defined we just
            # take a factor of one
            try:
                poisson_constraint_rate_factors.append(parset.factors)
            except AttributeError:
                # this seems to be dead code
                # TODO: add coverage (issue #540)
                poisson_constraint_rate_factors.append([1.0] * len(thisauxdata))

        if self.param_viewer.index_selection:
            self._poisson_data = default_backend.astensor(
                default_backend.concatenate(poisson_constraint_data), dtype='int'
            )

            _poisson_rate_fac = default_backend.astensor(
                default_backend.concatenate(poisson_constraint_rate_factors),
                dtype='float',
            )
            factors = default_backend.reshape(
                _poisson_rate_fac, (1, -1)
            )  # (1, normals)
            self._batched_factors = default_backend.tile(
                factors, (self.batch_size or 1, 1)
            )

            access_field = default_backend.concatenate(
                self.param_viewer.index_selection, axis=1
            )
            self._access_field = access_field

        else:
            self._poisson_data = None
            self._access_field = None
            self._batched_factors = None

        self._precompute()
        events.subscribe('tensorlib_changed')(self._precompute)

    def _precompute(self):
        if not self.param_viewer.index_selection:
            return
        tensorlib, _ = get_backend()
        self.poisson_data = tensorlib.astensor(self._poisson_data, dtype='int')
        self.access_field = tensorlib.astensor(self._access_field, dtype='int')
        self.batched_factors = tensorlib.astensor(self._batched_factors)

    def logpdf(self, auxdata, pars):
        tensorlib, _ = get_backend()
        if not self.param_viewer.index_selection:
            return (
                tensorlib.zeros(self.batch_size)
                if self.batch_size is not None
                else tensorlib.astensor(0.0)[0]
            )
        tensorlib, _ = get_backend()

        pars = tensorlib.astensor(pars)
        if self.batch_size == 1 or self.batch_size is None:
            batched_pars = tensorlib.reshape(
                pars, (self.batch_size or 1,) + tensorlib.shape(pars)
            )
        else:
            batched_pars = pars
        flat_pars = tensorlib.reshape(batched_pars, (-1,))
        nuispars = tensorlib.gather(flat_pars, self.access_field)

        pois_rates = tensorlib.product(
            tensorlib.stack([nuispars, self.batched_factors]), axis=0
        )

        # pdf pars are done, now get data and compute
        auxdata = tensorlib.astensor(auxdata)
        poisson_data = tensorlib.gather(auxdata, self.poisson_data)
        result = tensorlib.poisson_logpdf(poisson_data, pois_rates)
        result = tensorlib.sum(result, axis=1)
        if self.batch_size is None:
            return result[0]
        return result
