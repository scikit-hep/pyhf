from . import get_backend, default_backend

class gaussian_constraint_combined(object):
    def __init__(self,pdfconfig):
        tensorlib, _ = get_backend()
        self.tensorlib_name = tensorlib.name
        self.prepped = False
        # iterate over all constraints order doesn't matter....
        
        self.par_indices = list(range(len(pdfconfig.suggested_init())))
        self.data_indices = list(range(len(pdfconfig.auxdata)))
        self.mod_and_slice = [
            (pdfconfig.modifier(cname),pdfconfig.par_slice(cname))
            for cname in pdfconfig.auxdata_order
        ]

    def _precompute(self):
        tensorlib, _ = get_backend()
        # did things change that we need to recompute?
        tensor_type_change = tensorlib.name != self.tensorlib_name
        if (not tensor_type_change) and self.prepped:
            return
        start_index = 0        
        normal_constraint_data = []
        normal_constraint_mean_indices = []
        normal_constraint_sigmas = []
        for modifier,modslice in self.mod_and_slice:
            end_index = start_index + modifier.n_parameters
            thisauxdata = self.data_indices[start_index:end_index]
            start_index = end_index
            if not modifier.pdf_type == 'normal': continue

            # print('combined gauss: {}/{}/{}'.format(
            #     thisauxdata,par_indices[modslice],modifier.sigmas)
            # )
            try:
                normal_constraint_sigmas.append(modifier.sigmas)
            except AttributeError:
                normal_constraint_sigmas.append([1.]*len(thisauxdata))

            normal_constraint_data.append(thisauxdata)
            normal_constraint_mean_indices.append(self.par_indices[modslice])
        
        if normal_constraint_mean_indices:
            normal_mean_idc  = tensorlib.concatenate(list(map(lambda x: tensorlib.astensor(x,dtype = 'int'),normal_constraint_mean_indices)))
            normal_sigmas    = tensorlib.concatenate(list(map(tensorlib.astensor,normal_constraint_sigmas)))
            normal_data      = tensorlib.concatenate(list(map(lambda x: tensorlib.astensor(x,dtype = 'int'),normal_constraint_data)))
        else:
            normal_data, normal_sigmas, normal_mean_idc = None, None, None

        self.normal_data = normal_data
        self.normal_sigmas = normal_sigmas
        self.normal_mean_idc = normal_mean_idc
        self.prepped = True

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
        tensorlib, _ = get_backend()
        self.tensorlib_name = tensorlib.name
        self.prepped = False
        # iterate over all constraints order doesn't matter....

        self.par_indices = list(range(len(pdfconfig.suggested_init())))
        self.data_indices = list(range(len(pdfconfig.auxdata)))
        self.mod_and_slice = [
            (pdfconfig.modifier(cname),pdfconfig.par_slice(cname))
            for cname in pdfconfig.auxdata_order
        ]

    def _precompute(self):
        tensorlib, _ = get_backend()
        # did things change that we need to recompute?
        tensor_type_change = tensorlib.name != self.tensorlib_name
        if (not tensor_type_change) and self.prepped:
            return
        
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

            try:
                poisson_constraint_rate_factors.append(modifier.bkg_over_db_squared)
            except AttributeError:
                poisson_constraint_rate_factors.append(tensorlib.shape(self.par_indices[modslice]))


        if poisson_constraint_rate_indices:
            poisson_rate_idc  = tensorlib.concatenate(list(map(lambda x: tensorlib.astensor(x,dtype = 'int'), poisson_constraint_rate_indices)))
            poisson_rate_fac  = tensorlib.concatenate(list(map(lambda x: tensorlib.astensor(x,dtype = 'float'), poisson_constraint_rate_factors)))
            poisson_data      = tensorlib.concatenate(list(map(lambda x: tensorlib.astensor(x,dtype = 'int'), poisson_constraint_data)))
        else:
            poisson_rate_idc, poisson_data, poisson_rate_fac = None, None, None
        self.poisson_data = poisson_data
        self.poisson_rate_idc = poisson_rate_idc
        self.poisson_rate_fac = poisson_rate_fac
        self.prepped = True

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
