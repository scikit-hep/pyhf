from . import get_backend

class gaussian_constraint_combined(object):
    def __init__(self,pdf):
        tensorlib, _ = get_backend()
        # iterate over all constraints order doesn't matter....
        start_index = 0        
        normal_constraint_data = []
        normal_constraint_mean_indices = []
        normal_constraint_sigmas = []
        
        par_indices = list(range(len(pdf.config.suggested_init())))
        data_indices = list(range(len(pdf.config.auxdata)))
        for cname in pdf.config.auxdata_order:
            modifier = pdf.config.modifier(cname)
            modslice  = pdf.config.par_slice(cname)

            end_index = start_index + modifier.n_parameters
            thisauxdata = data_indices[start_index:end_index]
            start_index = end_index
            if not modifier.pdf_type == 'normal': continue

            # print('combined gauss: {}/{}/{}'.format(
            #     thisauxdata,par_indices[modslice],modifier.sigmas)
            # )
            normal_constraint_sigmas.append(modifier.sigmas)
            normal_constraint_data.append(thisauxdata)
            normal_constraint_mean_indices.append(par_indices[modslice])
        
        if normal_constraint_mean_indices:
            normal_mean_idc  = tensorlib.concatenate(map(lambda x: tensorlib.astensor(x,dtype = 'int'),normal_constraint_mean_indices))
            normal_sigmas    = tensorlib.concatenate(map(tensorlib.astensor,normal_constraint_sigmas))
            normal_data      = tensorlib.concatenate(map(lambda x: tensorlib.astensor(x,dtype = 'int'),normal_constraint_data))
        else:
            normal_data, normal_sigmas, normal_mean_idc = None, None, None

        self.prepped =  (normal_data,normal_sigmas,normal_mean_idc)

    def logpdf(self,auxdata,pars):
        if self.prepped[0] is None:
            return 0
        tensorlib, _ = get_backend()
        normal_data   = tensorlib.gather(auxdata,self.prepped[0])
        normal_means  = tensorlib.gather(pars,self.prepped[2])
        normal_sigmas = self.prepped[1]
        # for d,m,s in zip(normal_data,normal_means,normal_sigmas):
        #     print('fast data: {} mean: {} sigma: {}'.format(d,m,s))
        normal = tensorlib.normal_logpdf(normal_data,normal_means,normal_sigmas)
        return tensorlib.sum(normal)

class poisson_constraint_combined(object):
    def __init__(self,pdf):
        tensorlib, _ = get_backend()
        # iterate over all constraints order doesn't matter....
        start_index = 0
                        
        poisson_constraint_data = []
        poisson_constraint_rate_indices = []
        poisson_constraint_rate_factors = []

        par_indices = list(range(len(pdf.config.suggested_init())))
        data_indices = list(range(len(pdf.config.auxdata)))
        for cname in pdf.config.auxdata_order:
            modifier = pdf.config.modifier(cname)
            end_index = start_index + modifier.n_parameters
            thisauxdata = data_indices[start_index:end_index]
            start_index = end_index
            if not modifier.pdf_type == 'poisson': continue
            modslice  = pdf.config.par_slice(cname)

            poisson_constraint_data.append(thisauxdata)
            poisson_constraint_rate_indices.append(par_indices[modslice])

            try:
                poisson_constraint_rate_factors.append(modifier.bkg_over_db_squared)
            except AttributeError:
                poisson_constraint_rate_factors.append(tensorlib.shape(par_indices[modslice]))


        if poisson_constraint_rate_indices:
            poisson_rate_idc  = tensorlib.concatenate(map(lambda x: tensorlib.astensor(x,dtype = 'int'), poisson_constraint_rate_indices))
            poisson_rate_fac  = tensorlib.concatenate(map(lambda x: tensorlib.astensor(x,dtype = 'float'), poisson_constraint_rate_factors))
            poisson_data      = tensorlib.concatenate(map(lambda x: tensorlib.astensor(x,dtype = 'int'), poisson_constraint_data))
        else:
            poisson_rate_idc, poisson_data, poisson_rate_fac = None, None, None
        self.prepped = (poisson_data,poisson_rate_idc,poisson_rate_fac)

    def logpdf(self,auxdata,pars):
        if self.prepped[0] is None:
            return 0
        tensorlib, _ = get_backend()
        poisson_data  = tensorlib.gather(auxdata,self.prepped[0])
        poisson_rate_base  = tensorlib.gather(pars,self.prepped[1])
        poisson_factors  = self.prepped[2]

        poisson_rate = tensorlib.product(
            tensorlib.stack([poisson_rate_base, poisson_factors]), axis=0)
        # for d,r in zip(poisson_data,poisson_rate):
        #     print('fast data: {} rate: {}'.format(d,r))
        poisson = tensorlib.poisson_logpdf(poisson_data,poisson_rate)
        return tensorlib.sum(poisson)
