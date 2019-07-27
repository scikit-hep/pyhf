from . import get_backend, default_backend
from . import events
from .paramview import ParamViewer


class gaussian_constraint_combined(object):
    def __init__(self, pdfconfig, batch_size = 1):
        self.batch_size = batch_size
        # iterate over all constraints order doesn't matter....

        self.par_indices = list(range(len(pdfconfig.suggested_init())))
        self.data_indices = list(range(len(pdfconfig.auxdata)))
        self.parset_and_slice = [
            (pdfconfig.param_set(cname), pdfconfig.par_slice(cname))
            for cname in pdfconfig.auxdata_order
        ]

        pnames = [cname for cname in pdfconfig.auxdata_order if pdfconfig.param_set(cname).pdf_type == 'normal']

        parfield_shape = (self.batch_size, len(pdfconfig.suggested_init()))
        self.parameter_helper = ParamViewer(parfield_shape, pdfconfig.par_map, pnames, regular = False)

        # self.parameter_helper.index_selection  is [ (batch, parslice) ] ]
        access_field = None
        for x in self.parameter_helper.index_selection:
            access_field = x if access_field is None else default_backend.concatenate([access_field,x], axis=1)
        #access field is (nbatch, normals)
        self._access_field = access_field
        
        start_index = 0
        normal_constraint_data = []
        normal_constraint_sigmas = []
        for parset, parslice in self.parset_and_slice:
            end_index = start_index + parset.n_parameters
            thisauxdata = self.data_indices[start_index:end_index]
            start_index = end_index
            if not parset.pdf_type == 'normal':
                continue

            # many constraints are defined on a unit gaussian
            # but we reserved the possibility that a paramset
            # can define non-standard uncertainties. This is used
            # by the paramset associated to staterror modifiers.
            # Such parsets define a 'sigmas' attribute
            try:
                normal_constraint_sigmas.append(parset.sigmas)
            except AttributeError:
                normal_constraint_sigmas.append([1.0] * len(thisauxdata))

            normal_constraint_data.append(thisauxdata)


        if self.parameter_helper.index_selection:
            normal_sigmas = default_backend.concatenate(
                list(map(default_backend.astensor, normal_constraint_sigmas))
            )
            normal_data = default_backend.concatenate(
                list(
                    map(
                        lambda x: default_backend.astensor(x, dtype='int'),
                        normal_constraint_data,
                    )
                )
            )

            self._normal_data = default_backend.astensor(
                default_backend.tolist(normal_data), dtype='int'
            )
            self._normal_sigmas = default_backend.astensor(
                default_backend.tolist(normal_sigmas)
            )
        else:
            self._normal_data, self._normal_sigmas = (
                None,
                None,
            )

        sigmas = default_backend.reshape(self._normal_sigmas,(1,-1)) # (1, normals)
        self._batched_sigmas = default_backend.tile(sigmas,(self.batch_size,1))

        self._precompute()
        events.subscribe('tensorlib_changed')(self._precompute)

    def _precompute(self):
        if not self.parameter_helper.index_selection:
            return
        tensorlib, _ = get_backend()
        self.batched_sigmas = tensorlib.astensor(self._batched_sigmas)
        self.normal_data    = tensorlib.astensor(self._normal_data,dtype='int')
        self.normal_sigmas  = tensorlib.astensor(self._normal_sigmas)
        self.access_field    = tensorlib.astensor(self._access_field,dtype='int')

    def logpdf(self, auxdata, pars):
        tensorlib, _ = get_backend()
        if not self.parameter_helper.index_selection:
            return tensorlib.zeros(self.batch_size,)

        pars = tensorlib.astensor(pars)
        if self.batch_size == 1:
            batched_pars = tensorlib.reshape(
                pars, (self.batch_size,) + tensorlib.shape(pars)
            )
        else:
            batched_pars = pars


        auxdata = tensorlib.astensor(auxdata)
        normal_data = tensorlib.gather(auxdata, self.normal_data)
        flat_pars = tensorlib.reshape(batched_pars,(-1,))
        normal_means = tensorlib.gather(flat_pars, self.access_field)
        normal = tensorlib.normal_logpdf(normal_data, normal_means, self.batched_sigmas)
        result = tensorlib.sum(normal, axis = 1)
        return result


class poisson_constraint_combined(object):
    def __init__(self, pdfconfig, batch_size = 1):
        self.batch_size = batch_size
        # iterate over all constraints order doesn't matter....

        self.par_indices = list(range(len(pdfconfig.suggested_init())))
        self.data_indices = list(range(len(pdfconfig.auxdata)))
        self.mod_and_slice = [
            (pdfconfig.param_set(cname), pdfconfig.par_slice(cname))
            for cname in pdfconfig.auxdata_order
        ]
    
        pnames = [cname for cname in pdfconfig.auxdata_order if pdfconfig.param_set(cname).pdf_type == 'poisson']

        parfield_shape = (self.batch_size, len(pdfconfig.suggested_init()))
        self.parameter_helper = ParamViewer(parfield_shape, pdfconfig.par_map, pnames, regular = False)

        start_index = 0
        poisson_constraint_data = []
        poisson_constraint_rate_factors = []
        for parset, parslice in self.mod_and_slice:
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
                poisson_constraint_rate_factors.append(
                    default_backend.shape(self.par_indices[parslice])
                )

        if self.parameter_helper.index_selection:
            poisson_rate_fac = default_backend.concatenate(
                list(
                    map(
                        lambda x: default_backend.astensor(x, dtype='float'),
                        poisson_constraint_rate_factors,
                    )
                )
            )
            poisson_data = default_backend.concatenate(
                list(
                    map(
                        lambda x: default_backend.astensor(x, dtype='int'),
                        poisson_constraint_data,
                    )
                )
            )

            self._poisson_data = default_backend.astensor(
                default_backend.tolist(poisson_data), dtype='int'
            )
            self.poisson_rate_fac = default_backend.astensor(
                default_backend.tolist(poisson_rate_fac), dtype='float'
            )
        else:
            self._poisson_data, self.poisson_rate_fac = (
                None,
                None,
            )


        # self.parameter_helper.index_selection  is [ (batch, parslice) ] ]
        access_field = None
        for x in self.parameter_helper.index_selection:
            access_field = x if access_field is None else default_backend.concatenate([access_field,x], axis=1)
        #access field is (nbatch, normals)
        self._access_field = access_field


        factors = default_backend.reshape(self.poisson_rate_fac,(1,-1)) # (1, normals)
        self._batched_factors = default_backend.tile(factors,(self.batch_size,1))

        self._precompute()
        events.subscribe('tensorlib_changed')(self._precompute)

    def _precompute(self):
        if not self.parameter_helper.index_selection:
            return
        tensorlib, _ = get_backend()
        self.poisson_data    = tensorlib.astensor(self._poisson_data,dtype='int')
        self.access_field    = tensorlib.astensor(self._access_field,dtype='int')
        self.batched_factors = tensorlib.astensor(self._batched_factors)
    
    def logpdf(self, auxdata, pars):
        tensorlib, _ = get_backend()
        if not self.parameter_helper.index_selection:
            return tensorlib.zeros(self.batch_size,)
        tensorlib, _ = get_backend()
        auxdata = tensorlib.astensor(auxdata)
        poisson_data = tensorlib.gather(auxdata, self.poisson_data)

        pars = tensorlib.astensor(pars)
        if self.batch_size == 1:
            batched_pars = tensorlib.reshape(
                pars, (self.batch_size,) + tensorlib.shape(pars)
            )
        else:
            batched_pars = pars
        flat_pars = tensorlib.reshape(batched_pars,(-1,))
        nuispars  = tensorlib.gather(flat_pars,self.access_field)


        pois_rates = tensorlib.product(tensorlib.stack([nuispars, self.batched_factors]), axis=0)

        result = tensorlib.poisson_logpdf(poisson_data,pois_rates)
        result = tensorlib.sum(result, axis = 1)
        return result