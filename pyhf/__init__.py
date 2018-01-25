import numpy as np
from scipy.stats import poisson, norm
from scipy.optimize import minimize

def _poisson_impl(n, lam):
    #use continuous gaussian approx, b/c asimov data may not be integer
    return norm.pdf(n, loc = lam, scale = np.sqrt(lam))

@np.vectorize
def _multiply_arrays_or_scalars(*args):
    '''
    calling this function with arrays (a and b) and/or scalers (s1,s2) it will broadcast as usua
    result is a array with the row-wise products
    r1 = | a1 | * s1 * | b1 | * s2
    r2 = | a2 | * s1 * | b2 | * s2
    r3 = | a3 | * s1 * | b3 | * s2
    '''
    return np.prod(np.array(args))

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
        self.nuis_map    = {}
        self.next_index = 0

    def all_mods(self, only_constraints = False):
        if not only_constraints: return self.nuis_map.keys()
        return [k for k,v in self.nuis_map.items() if v['mod'] is not None]

    def par_slice(self,name):
        return self.nuis_map[name]['slice']

    def mod(self,name):
        return self.nuis_map[name]['mod']

    def add_mod(self,name, npars, mod):
        sl = slice(self.next_index, self.next_index + npars)
        self.next_index = self.next_index + npars
        self.nuis_map[name] = {
            'slice': sl,
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
                        'type': 'normfactor',
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
        #hacky, need to keep track in which order we added the constraints
        #so that we can generate correctly-ordered data
        self.auxdata_order = []
        for sample, sample_def in self.samples.items():
            for mod_def in sample_def['mods']:
                mod = None
                if mod_def['type'] == 'normfactor':
                    mod = None # no object for factors
                    self.config.add_mod(mod_def['name'],npars = 1,mod = mod)
                if mod_def['type'] == 'shapesys':
                    #we reserve one parameter for each bin
                    mod = shapesys_constraint(sample_def['data'],mod_def['data'])
                    self.config.add_mod(mod_def['name'],npars = len(sample_def['data']), mod = mod)

                     #it's a constraint, so this implies more data
                    self.auxdata += self.config.mod(mod_def['name']).auxdata
                    self.auxdata_order.append(mod_def['name'])

    def _multiplicative_factors(self,sample,pars):
        multiplicative_types = ['shapesys','normfactor']
        mods = [m['name'] for m in self.samples[sample]['mods'] if m['type'] in multiplicative_types]
        return [pars[self.config.par_slice(m)] for m in mods]

    def expected_sample(self,name,pars):
        facs = self._multiplicative_factors(name,pars)
        nom  = self.samples[name]['data']
        factors = facs + [nom]
        return _multiply_arrays_or_scalars(*factors)

    def expected_auxdata(self, pars):
        ### probably more correctly this should be the expectation value of the constraint_pdf
        ### or for the constraints we are using (single par constraings with mean == mode), we can
        ### just return the alphas

        ### order matters! because we generated auxdata in a certain order
        for modname in self.auxdata_order:
            return self.config.mod(modname).expected_data(pars[self.config.par_slice(modname)])

    def expected_actualdata(self, pars):
        counts = [self.expected_sample(sname,pars) for sname in self.samples]
        return [sum(sample_counts) for sample_counts in zip(*counts)]

    def expected_data(self, pars, include_auxdata = True):

        expected_actual = self.expected_actualdata(pars)
        if not include_auxdata:
            return expected_actual

        expected_constraints = self.expected_auxdata(pars)
        return expected_actual + expected_constraints

    def constraint_pdf(self,auxdata, pars):
        product = 1.0
        #iterate over all constraints order doesn't matter....
        for cname in self.config.all_mods(only_constraints=True):
            mod, modslice = self.config.mod(cname), self.config.par_slice(cname)
            modalphas = mod.alphas(pars[modslice])
            for a,alpha in zip(auxdata, modalphas):
                product = product * mod.pdf(a, alpha)
        return product

    def pdf(self, pars, data):
        # split data in actual data and aux_data (inputs into constraints)
        cut = len(data) - len(self.auxdata)
        actual_data, aux_data = data[:cut], data[cut:]

        product = 1

        lambdas_data = self.expected_actualdata(pars)
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
