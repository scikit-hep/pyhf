import numpy as np
from scipy.stats import poisson, norm
from scipy.optimize import minimize

def _poisson_impl(n, lam):
    #use continuous gaussian approx, b/c asimov data may not be integer
    return norm.pdf(n, loc = lam, scale = np.sqrt(lam))

class shapesys_constraint(object):
    def __init__(self, nom_data, mod_data):
        self.auxdata = []
        self.bkg_over_db_squared = []
        for b, deltab in zip(nom_data, mod_data):
            bkg_over_bsq = b*b/deltab/deltab # tau*b
            self.bkg_over_db_squared.append(bkg_over_bsq)
            self.auxdata.append(bkg_over_bsq)

    def alphas(self,pars):
        return [gamma * c for gamma, c in zip(pars, self.bkg_over_db_squared)]

    def pdf(self, a, alpha):
        return _poisson_impl(a, alpha)

    def expected_data(self,pars):
        return self.alphas(pars)

class modelconfig(object):
    def __init__(self):
        self.poi_index = 0
        self.nuis_map = {}

    def par_slice(self,name):
        return self.nuis_map[name]['slice']

    def mod(self,name):
        return self.nuis_map[name]['mod']

    def add_mod(self,name, slice, mod):
        self.nuis_map[name] = {
            'slice': slice,
            'mod': mod
        }

class hfpdf(object):
    def __init__(self, signal_data, bkg_data, bkg_uncerts):
        self.config = modelconfig()

        self.samples = {
            'signal': {
                'data': signal_data,
                'mods': [
                    {
                        'name': 'mu',
                        'type': 'normfac',
                        'data': None
                    }
                ]
            },
            'background': {
                'data': bkg_data,
                'mods': [
                    {
                        'name': 'uncorr_bkguncrt',
                        'type': 'shapesys',
                        'data': bkg_uncerts
                    }
                ]
            }
        }
        self.auxdata   = []
        self.config.add_mod('bkg_shape_sys',slice(1,None),
            shapesys_constraint(
                self.samples['background']['data'],
                self.samples['background']['mods'][0]['data']
            )
        )
        self.auxdata += self.config.mod('bkg_shape_sys').auxdata

    def expected_signal(self, pars):
        poi = pars[0]
        nominal_signals = self.samples['signal']['data']
        return [poi*snom for snom in nominal_signals]

    def expected_background(self,pars):
        nom = self.samples['background']['data']      #nominal histo
        sl  = self.config.par_slice('bkg_shape_sys')  #bin-wise factors
        return [gamma * b0 for gamma,b0 in zip(pars[sl],nom)]

    def expected_auxdata(self, pars):
        ### probably more correctly this should be the expectation value of the constraint_pdf
        ### or for the constraints we are using (single par constraings with mean == mode), we can
        ### just return the alphas

        ### need to figure out how to iterate all nuis pars in order
        modname = 'bkg_shape_sys'
        return self.config.mod(modname).expected_data(pars[self.config.par_slice(modname)])

    def expected_actualdata(self, pars):
        signal_counts     = self.expected_signal(pars)
        background_counts = self.expected_background(pars)
        return [s+b for s,b in zip(signal_counts, background_counts)]

    def expected_data(self, pars, include_auxdata = True):

        expected_actual = self.expected_actualdata(pars)
        if not include_auxdata:
            return expected_actual

        expected_constraints = self.expected_auxdata(pars)
        return expected_actual + expected_constraints

    def constraint_pdf(self,auxdata, pars):
        product = 1.0
        #iterate over all constraints
        for cname in ['bkg_shape_sys']:
            mod, modslice = self.config.mod(cname), self.config.par_slice(cname)
            modalphas = mod.alphas(pars[modslice])
            for a,alpha in zip(auxdata, modalphas):
                product = product * mod.pdf(a, alpha)
        return product

    def pdf(self, pars, data):
        lambdas_data = self.expected_actualdata(pars)

        # split data in actual data and aux_data (inputs into constraints)
        actual_data,aux_data = data[:len(lambdas_data)], data[len(lambdas_data):]

        product = 1
        for d,lam in zip(actual_data, lambdas_data):
            product = product * _poisson_impl(d, lam)

        product = product * self.constraint_pdf(aux_data, pars)
        return product


def generate_asimov_data(asimov_mu, data, pdf, init_pars,par_bounds):
    bestfit_nuisance_asimov = constrained_bestfit(asimov_mu,data, pdf, init_pars,par_bounds)
    return pdf.expected_data(bestfit_nuisance_asimov)

##########################

def loglambdav(pars, data, pdf):
    return -2*np.log(pdf.pdf(pars, data))

### The Test Statistic
def qmu(mu,data, pdf, init_pars,par_bounds):
    mubhathat = constrained_bestfit(mu,data,pdf, init_pars,par_bounds)
    muhatbhat = unconstrained_bestfit(data,pdf, init_pars,par_bounds)
    qmu = loglambdav(mubhathat, data,pdf)-loglambdav(muhatbhat, data,pdf)
    if muhatbhat[0] > mu:
        return 0.0
    if -1e-6 < qmu <0:
        print 'WARNING: qmu negative: ', qmu
        return 0.0
    return qmu

### The Global Fit
def unconstrained_bestfit(data,pdf, init_pars,par_bounds):
    result = minimize(loglambdav, init_pars, method='SLSQP', args = (data,pdf), bounds = par_bounds)
    try:
        assert result.success
    except AssertionError:
        print result
    return result.x

### The Fit Conditions on a specific POI value
def constrained_bestfit(constrained_mu,data,pdf, init_pars,par_bounds):
    cons = {'type': 'eq', 'fun': lambda v: v[0]-constrained_mu}
    result = minimize(loglambdav, init_pars, constraints=cons, method='SLSQP',args = (data,pdf), bounds = par_bounds)
    try:
        assert result.success
    except AssertionError:
        print result
    return result.x

def pvals_from_teststat(sqrtqmu_v,sqrtqmuA_v):
    CLsb = 1-norm.cdf(sqrtqmu_v )
    CLb  = norm.cdf(sqrtqmuA_v-sqrtqmu_v)
    CLs  = CLb/CLsb
    return CLsb,CLb,CLs

def runOnePoint(muTest, data,pdf,init_pars,par_bounds):
    asimov_mu = 0.0
    asimov_data = generate_asimov_data(asimov_mu,data,pdf,init_pars,par_bounds)

    qmu_v  = qmu(muTest,data,pdf, init_pars,par_bounds)
    qmuA_v = qmu(muTest,asimov_data,pdf,init_pars,par_bounds)
    sqrtqmu_v = np.sqrt(qmu_v)
    sqrtqmuA_v = np.sqrt(qmuA_v)

    sigma = muTest/sqrtqmuA_v if sqrtqmuA_v > 0 else None

    CLsb,CLb,CLs = pvals_from_teststat(sqrtqmu_v, sqrtqmuA_v)

    CLs_exp = []
    for nsigma in [-2,-1,0,1,2]:
        sqrtqmu_v_sigma =  sqrtqmuA_v-nsigma
        CLs_exp.append(pvals_from_teststat(sqrtqmu_v_sigma,sqrtqmuA_v)[-1])
    return qmu_v,qmuA_v,sigma,CLsb,CLb,CLs,CLs_exp
