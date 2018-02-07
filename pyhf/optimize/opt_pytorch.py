import torch.optim
class pytorch_optimizer(object):
    def __init__(self, **kwargs):
        self.tensorlib = kwargs['tensorlib']
        self.maxdelta = kwargs.get('maxdela',1e-4)

    def unconstrained_bestfit(self, objective, data, pdf, init_pars, par_bounds):
        init_pars = self.tensorlib.astensor(init_pars)
        init_pars.requires_grad = True
        optimizer = torch.optim.Adam([init_pars])
        maxdelta = None
        for i in range(10000):
            loss = objective(init_pars,data,pdf)
            optimizer.zero_grad()
            loss.backward()
            init_old = init_pars.data.clone()
            optimizer.step()
            maxdelta = (init_pars.data - init_old).abs().max()
            if maxdelta < self.maxdelta:
                break
        return init_pars

    def constrained_bestfit(self, objective, constrained_mu, data, pdf, init_pars, par_bounds):
        allvars = [self.tensorlib.astensor([v] if i!= pdf.config.poi_index else [constrained_mu]) for i,v in enumerate(init_pars)]
        nuis_pars = [v for i,v in enumerate(allvars) if i != pdf.config.poi_index]
        for np in nuis_pars: np.requires_grad = True
        poi_par   = [v for i,v in enumerate(allvars) if i == pdf.config.poi_index][0]

        def assemble(poi_par, nuis_pars):
            pars = [x for x in nuis_pars]
            pars.insert(pdf.config.poi_index,poi_par)
            pars = self.tensorlib.concatenate(pars)
            return pars

        optimizer = torch.optim.Adam(nuis_pars)
        for i in range(10000):
            pars = assemble(poi_par, nuis_pars)
            loss = objective(pars,data,pdf)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            after_pars = assemble(poi_par, nuis_pars)
            maxdelta = (after_pars.data - pars.data).abs().max()
            if maxdelta < self.maxdelta:
                break
        return pars
