from . import get_backend, default_backend


class standard_gaussian_constraint(object):
    def __init__(self, n_parameters, inits, bounds, auxdata):
        self.pdf_type = 'normal'
        self.n_parameters = n_parameters
        self.suggested_init = inits
        self.suggested_bounds = bounds
        self.auxdata = auxdata

    def alphas(self, pars):
        return pars  # the nuisance parameters correspond directly to the alpha

    def expected_data(self, pars):
        return self.alphas(pars)

class factor_poisson_constraint(object):
    def __init__(self, n_parameters, inits, bounds, auxdata, factors):
        self.factors = factors
        self.pdf_type = 'poisson'
        self.n_parameters = n_parameters
        self.suggested_init = inits
        self.suggested_bounds = bounds
        self.auxdata = auxdata

    def alphas(self, pars):
        tensorlib, _ = get_backend()
        return tensorlib.product(tensorlib.stack([pars, tensorlib.astensor(self.factors)]), axis=0)
    def expected_data(self, pars):
        return self.alphas(pars)


class gaussian_constraint_combined(object):
    def __init__(self,pdfconfig):
        # iterate over all constraints order doesn't matter....
        
        self.par_indices = list(range(len(pdfconfig.suggested_init())))
        self.data_indices = list(range(len(pdfconfig.auxdata)))
        self.mod_and_slice = [
            (pdfconfig.modifier(cname),pdfconfig.par_slice(cname))
            for cname in pdfconfig.auxdata_order
        ]
        self.tensorlib_name = None
        self._precompute()
        tensorlib, _ = get_backend()

    def _precompute(self):
        tensorlib, _ = get_backend()
        # did things change that we need to recompute?
        tensor_type_change = tensorlib.name != self.tensorlib_name
        if not tensor_type_change:
            return
        self.tensorlib_name = tensorlib.name
        start_index = 0        
        normal_constraint_data = []
        normal_constraint_mean_indices = []
        normal_constraint_sigmas = []
        for modifier,modslice in self.mod_and_slice:
            end_index = start_index + modifier.n_parameters
            thisauxdata = self.data_indices[start_index:end_index]
            start_index = end_index
            if not modifier.pdf_type == 'normal': continue

            # many constraints are defined on a unit gaussian
            # but we reserved the possibility that a modifier
            # can define its own uncertainties. This is used
            # by the staterror modifier. Such modifiers
            # are required to define a 'sigmas' attribute
            try:
                normal_constraint_sigmas.append(modifier.sigmas)
            except AttributeError:
                normal_constraint_sigmas.append([1.]*len(thisauxdata))

            normal_constraint_data.append(thisauxdata)
            normal_constraint_mean_indices.append(self.par_indices[modslice])
        
        if normal_constraint_mean_indices:
            normal_mean_idc  = default_backend.concatenate(list(map(lambda x: default_backend.astensor(x,dtype = 'int'),normal_constraint_mean_indices)))
            normal_sigmas    = default_backend.concatenate(list(map(default_backend.astensor,normal_constraint_sigmas)))
            normal_data      = default_backend.concatenate(list(map(lambda x: default_backend.astensor(x,dtype = 'int'),normal_constraint_data)))

            self.normal_data = tensorlib.astensor(default_backend.tolist(normal_data),dtype = 'int')
            self.normal_sigmas = tensorlib.astensor(default_backend.tolist(normal_sigmas))
            self.normal_mean_idc = tensorlib.astensor(default_backend.tolist(normal_mean_idc),dtype = 'int')
        else:
            self.normal_data, self.normal_sigmas, self.normal_mean_idc = None, None, None

    def logpdf(self,auxdata,pars):
        self._precompute()
        if self.normal_data is None:
            return 0
        tensorlib, _ = get_backend()
        normal_data   = tensorlib.gather(auxdata,self.normal_data)
        normal_means  = tensorlib.gather(pars,self.normal_mean_idc)
        normal = tensorlib.normal_logpdf(normal_data,normal_means,self.normal_sigmas)
        return tensorlib.sum(normal)

class poisson_constraint_combined(object):
    def __init__(self,pdfconfig):
        # iterate over all constraints order doesn't matter....

        self.par_indices = list(range(len(pdfconfig.suggested_init())))
        self.data_indices = list(range(len(pdfconfig.auxdata)))
        self.mod_and_slice = [
            (pdfconfig.modifier(cname),pdfconfig.par_slice(cname))
            for cname in pdfconfig.auxdata_order
        ]
        self.tensorlib_name = None
        self._precompute()
        tensorlib, _ = get_backend()

    def _precompute(self):
        tensorlib, _ = get_backend()
        # did things change that we need to recompute?
        tensor_type_change = tensorlib.name != self.tensorlib_name
        if not tensor_type_change:
            return
        self.tensorlib_name = tensorlib.name
        
        start_index = 0
        poisson_constraint_data = []
        poisson_constraint_rate_indices = []
        poisson_constraint_rate_factors = []
        for modifier,modslice in self.mod_and_slice:
            end_index = start_index + modifier.n_parameters
            thisauxdata = self.data_indices[start_index:end_index]
            start_index = end_index
            if not modifier.pdf_type == 'poisson': continue

            poisson_constraint_data.append(thisauxdata)
            poisson_constraint_rate_indices.append(self.par_indices[modslice])

            # poisson constraints can specify a scaling factor for the 
            # backgrounds rates (see: on-off problem with a aux measurement
            # with tau*b). If such a scale factor is not defined we just
            # take a factor of one
            try:
                poisson_constraint_rate_factors.append(modifier.bkg_over_db_squared)
            except AttributeError:
                poisson_constraint_rate_factors.append(default_backend.shape(self.par_indices[modslice]))


        if poisson_constraint_rate_indices:
            poisson_rate_idc  = default_backend.concatenate(list(map(lambda x: default_backend.astensor(x,dtype = 'int'), poisson_constraint_rate_indices)))
            poisson_rate_fac  = default_backend.concatenate(list(map(lambda x: default_backend.astensor(x,dtype = 'float'), poisson_constraint_rate_factors)))
            poisson_data      = default_backend.concatenate(list(map(lambda x: default_backend.astensor(x,dtype = 'int'), poisson_constraint_data)))

            self.poisson_data = tensorlib.astensor(default_backend.tolist(poisson_data),dtype = 'int')
            self.poisson_rate_idc = tensorlib.astensor(default_backend.tolist(poisson_rate_idc),dtype = 'int')
            self.poisson_rate_fac = tensorlib.astensor(default_backend.tolist(poisson_rate_fac),dtype = 'float')
        else:
            self.poisson_rate_idc, self.poisson_data, self.poisson_rate_fac = None, None, None

    def logpdf(self,auxdata,pars):
        self._precompute()
        if self.poisson_data is None:
            return 0
        tensorlib, _ = get_backend()
        poisson_data  = tensorlib.gather(auxdata,self.poisson_data)
        poisson_rate_base  = tensorlib.gather(pars,self.poisson_rate_idc)
        poisson_factors  = self.poisson_rate_fac

        poisson_rate = tensorlib.product(
            tensorlib.stack([poisson_rate_base, poisson_factors]), axis=0)
        poisson = tensorlib.poisson_logpdf(poisson_data,poisson_rate)
        return tensorlib.sum(poisson)
