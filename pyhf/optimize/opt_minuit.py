import iminuit
import logging
import numpy as np

log = logging.getLogger(__name__)

class minuit_optimizer(object):
    def __init__(self):
        pass

    def _make_minuit(self, objective, data, pdf, init_pars, init_bounds, constrained_mu = None):
        def f(pars):
            result = objective(pars,data,pdf)
            logpdf = result[0]
            return logpdf
        parnames =  ['p{}'.format(i) for i in range(len(init_pars))]
        kw = {'limit_p{}'.format(i): b for i,b in enumerate(init_bounds)}
        initvals = {'p{}'.format(i): v for i,v in enumerate(init_pars)}
        if constrained_mu is not None:
            constraints = {'fix_p{}'.format(pdf.config.poi_index): True}
            initvals['p{}'.format(pdf.config.poi_index)] = constrained_mu
        else:
            constraints = {}
        mm = iminuit.Minuit(f, use_array_call=True, forced_parameters = parnames, **kw, **constraints, **initvals)
        return mm

    def unconstrained_bestfit(self, objective, data, pdf, init_pars, par_bounds):
        # The Global Fit
        mm = self._make_minuit(objective,data,pdf,init_pars,par_bounds)
        mm.migrad()
        return np.asarray([x[1] for x in mm.values.items()])

    def constrained_bestfit(self, objective, constrained_mu, data, pdf, init_pars, par_bounds):
        # The Fit Conditions on a specific POI value
        mm = self._make_minuit(objective,data,pdf,init_pars,par_bounds, constrained_mu = constrained_mu)
        mm.migrad()
        return np.asarray([x[1] for x in mm.values.items()])
