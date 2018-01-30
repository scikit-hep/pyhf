import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import logging

log = logging.getLogger(__name__)

def _poisson_impl(n, lam):
    #use continuous gaussian approx, b/c asimov data may not be integer
    # print 'pois', n,lam
    # return poisson.pmf(n,lam)
    # print 'lam,sqrtlam',lam,np.sqrt(lam)
    return norm.pdf(n, loc = lam, scale = np.sqrt(lam))

def _gaussian_impl(x, mu, sigma):
    #use continuous gaussian approx, b/c asimov data may not be integer
    # print 'gaus', x, mu, sigma
    return norm.pdf(x, loc = mu, scale = sigma)

def _hfinterp_code0(at_minus_one, at_zero, at_plus_one):
    @np.vectorize
    def func(alpha):
        if alpha >  0   : return (at_plus_one-at_zero)*alpha
        if alpha <= 0   : return (at_zero-at_minus_one)*alpha
    return func

def _hfinterp_code1(at_minus_one, at_zero, at_plus_one):
    @np.vectorize
    def func(alpha):
        if alpha >  0   : return pow(at_plus_one/at_zero, alpha)
        if alpha <= 0   : return pow(at_minus_one/at_zero, -alpha)
    return func

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

class normsys_constraint(object):
    def __init__(self, nom_data, mod_data):
        self.at_zero      = 1
        self.at_minus_one = mod_data['lo']
        self.at_plus_one  = mod_data['hi']
        self.auxdata = [0] #observed data is always at a = 1

    def alphas(self,pars):
        return pars #the nuisance parameters correspond directly to the alpha

    def expected_data(self,pars):
        return self.alphas(pars)

    def pdf(self, a, alpha):
        # print 'normsys gaussian'
        return _gaussian_impl(a, alpha , 1)


class histosys_constraint(object):
    def __init__(self, nom_data, mod_data):
        self.at_zero      = nom_data
        self.at_minus_one = mod_data['lo_hist']
        self.at_plus_one  = mod_data['hi_hist']
        self.auxdata = [0] #observed data is always at a = 1

    def alphas(self,pars):
        return pars #the nuisance parameters correspond directly to the alpha

    def expected_data(self,pars):
        return self.alphas(pars)

    def pdf(self, a, alpha):
        return _gaussian_impl(a, alpha , 1)

class shapesys_constraint(object):
    def __init__(self, nom_data, mod_data):
        self.auxdata = []
        self.bkg_over_db_squared = []
        for b, deltab in zip(nom_data, mod_data):
            bkg_over_bsq = b*b/deltab/deltab # tau*b
            log.info('shapesys for b,delta b (%s, %s) -> tau*b = %s', b, deltab, bkg_over_bsq)
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
        self.poi_index = None
        self.par_map    = {}
        self.next_index = 0
        self.snapshots  = {}
        self.par_order  = []

    def all_mods(self, only_constraints = False):
        if not only_constraints: return self.par_map.keys()
        return [k for k,v in self.par_map.items() if v['mod'] is not None]

    def suggested_init(self):
        init = []
        for name in self.par_order:
            init = init + self.par_map[name]['suggested_init']
        return init

    def suggested_bounds(self):
        bounds = []
        for name in self.par_order:
            bounds = bounds + self.par_map[name]['suggested_bounds']
        return bounds

    def par_slice(self,name):
        return self.par_map[name]['slice']

    def mod(self,name):
        return self.par_map[name]['mod']

    def add_mod(self,name, npars, mod, suggested_init, suggested_bounds):
        sl = slice(self.next_index, self.next_index + npars)
        self.next_index = self.next_index + npars
        if name in self.par_map:
            raise RuntimeError('shared systematic not implemented yet (processing {})'.format(name))
        log.info('adding modifier %s (%s new nuisance parameters)', name, npars)
        self.par_order.append(name)
        self.par_map[name] = {
            'slice': sl,
            'mod': mod,
            'suggested_init': suggested_init,
            'suggested_bounds': suggested_bounds
        }

class hfpdf(object):
    def __init__(self, channels):
        self.channels = channels
        self.config = modelconfig()
        self.auxdata   = []
        #hacky, need to keep track in which order we added the constraints
        #so that we can generate correctly-ordered data
        self.auxdata_order = []
        self.channel_order = []
        for ch, samples in self.channels.items():
            self.channel_order.append(ch)
            for sample, sample_def in samples.items():
                for mod_def in sample_def['mods']:
                    if mod_def['type'] == 'normfactor':
                        mod = None # no object for factors
                        self.config.add_mod(
                            name = mod_def['name'],
                            mod  = mod,
                            npars = 1,
                            suggested_init   = [1.0],
                            suggested_bounds = [[0,10]]
                        )
                        self.config.poi_index = self.config.par_slice(mod_def['name']).start
                    if mod_def['type'] == 'shapesys':
                        #we reserve one parameter for each bin
                        mod = shapesys_constraint(sample_def['data'],mod_def['data'])
                        self.config.add_mod(
                            name  = mod_def['name'],
                            npars = len(sample_def['data']),
                            suggested_init   = [1.0] * len(sample_def['data']),
                            suggested_bounds = [[0,10]] * len(sample_def['data']),
                            mod = mod,
                        )
                        self.auxdata += self.config.mod(mod_def['name']).auxdata
                        self.auxdata_order.append(mod_def['name'])
                    if mod_def['type'] == 'normsys':
                        mod = normsys_constraint(sample_def['data'],mod_def['data'])
                        self.config.add_mod(
                            name = mod_def['name'],
                            npars = 1,
                            mod = mod,
                            suggested_init   = [1.0],
                            suggested_bounds = [[-5,5]]
                        )
                        self.auxdata += self.config.mod(mod_def['name']).auxdata
                        self.auxdata_order.append(mod_def['name'])
                    if mod_def['type'] == 'histosys':
                        mod = histosys_constraint(sample_def['data'],mod_def['data'])
                        self.config.add_mod(
                            mod_def['name'],
                            npars = 1,
                            mod = mod,
                            suggested_init   = [1.0],
                            suggested_bounds = [[-5,5]])
                        self.auxdata += self.config.mod(mod_def['name']).auxdata
                        self.auxdata_order.append(mod_def['name'])

    def _multiplicative_factors(self,channel,sample,pars):
        multiplicative_types = ['shapesys','normfactor']
        mods = [m['name'] for m in self.channels[channel][sample]['mods'] if m['type'] in multiplicative_types]
        return [pars[self.config.par_slice(m)] for m in mods]

    def _normsysfactor(self, channel, sample, pars):
        # normsysfactor(nom_sys_alphas)   = 1 + sum(interp(1, anchors[i][0], anchors[i][0], val=alpha)  for i in range(nom_sys_alphas))
        mods = [m['name'] for m in self.channels[channel][sample]['mods'] if m['type'] == 'normsys']
        factor =  1

        for m in mods:
            mod = self.config.mod(m)
            val = pars[self.config.par_slice(m)]
            assert len(val)==1
            interp = _hfinterp_code1(mod.at_minus_one,mod.at_zero,mod.at_plus_one)
            factor = factor * interp(val[0])
        return factor

    def _histosysdelta(self, channel, sample, pars):
        mods = [m['name'] for m in self.channels[channel][sample]['mods'] if m['type'] == 'histosys']
        deltas = []
        for m in mods:
            mod = self.config.mod(m)
            val = pars[self.config.par_slice(m)]
            assert len(val)==1
            #intepolate for each bin
            mod_delta = []
            for lo_bin, nom_bin, up_bin, in zip(mod.at_minus_one,mod.at_zero,mod.at_plus_one):

                interp = _hfinterp_code0(lo_bin,nom_bin,up_bin)

                interp_val = float(interp(val[0]))
                mod_delta.append(interp_val)

            deltas = np.sum([deltas,mod_delta],axis=0)
            # print 'MOD DELTA',m,mod_delta

        return deltas

    def expected_sample(self,channel,sample,pars):
        # for each sample the expected ocunts are
        # counts = (multiplicative factors) * (normsys multiplier) * (histsys delta + nominal hist)
        #        = f1*f2*f3*f4* nomsysfactor(nom_sys_alphas) * hist(hist_addition(histosys_alphas) + nomdata)
        # nomsysfactor(nom_sys_alphas)   = 1 + sum(interp(1, anchors[i][0], anchors[i][0], val=alpha)  for i in range(nom_sys_alphas))
        # hist_addition(histosys_alphas) = sum(interp(nombin, anchors[i][0], anchors[i][0], val=alpha) for i in range(histosys_alphas))
        nom  = self.channels[channel][sample]['data']
        histosys_delta = self._histosysdelta(channel,sample, pars)
        interp_histo   = np.sum([nom,histosys_delta], axis=0)

        factors = []
        factors += self._multiplicative_factors(channel,sample,pars)
        factors += [self._normsysfactor(channel,sample, pars)]
        factors += [interp_histo]
        return _multiply_arrays_or_scalars(*factors)

    def expected_auxdata(self, pars):
        ### probably more correctly this should be the expectation value of the constraint_pdf
        ### or for the constraints we are using (single par constraings with mean == mode), we can
        ### just return the alphas

        ### order matters! because we generated auxdata in a certain order
        auxdata = []
        for modname in self.auxdata_order:
            thisaux = self.config.mod(modname).expected_data(pars[self.config.par_slice(modname)])
            auxdata = np.append(auxdata,thisaux)
        return auxdata

    def expected_actualdata(self, pars):
        data = []
        for channel in self.channel_order:
            counts = [self.expected_sample(channel,sample,pars) for sample in self.channels[channel]]
            data  += [sum(sample_counts) for sample_counts in zip(*counts)]
        return data

    def expected_data(self, pars, include_auxdata = True):
        expected_actual = self.expected_actualdata(pars)

        if not include_auxdata:
            return expected_actual
        expected_constraints = self.expected_auxdata(pars)
        return np.concatenate([expected_actual, expected_constraints])

    def constraint_logpdf(self,auxdata, pars):
        sum = 0
        #iterate over all constraints order doesn't matter....
        start_index = 0
        for cname in self.auxdata_order:
            # print 'ADDING constraint pdf term for ', cname
            mod, modslice = self.config.mod(cname), self.config.par_slice(cname)
            modalphas = mod.alphas(pars[modslice])
            end_index = start_index+len(modalphas)
            thisauxdata = auxdata[start_index:end_index]
            start_index = end_index
            # print 'a / alphas {} | {}'.format(thisauxdata, modalphas)
            for a,alpha in zip(thisauxdata, modalphas):
                sum = sum + np.log(mod.pdf(a, alpha))
        return sum

    def logpdf(self, pars, data):
        cut = len(data) - len(self.auxdata)
        actual_data, aux_data = data[:cut], data[cut:]

        sum = 0

        lambdas_data = self.expected_actualdata(pars)
        # print 'data / lamds {} | {}'.format(actual_data, lambdas_data)
        for d,lam in zip(actual_data, lambdas_data):
            sum = sum + np.log(_poisson_impl(d, lam))

        sum = sum + self.constraint_logpdf(aux_data, pars)
        return sum

    def pdf(self, pars, data):
        return np.exp(self.logpdf(pars, data))

def generate_asimov_data(asimov_mu, data, pdf, init_pars,par_bounds):
    bestfit_nuisance_asimov = constrained_bestfit(asimov_mu,data, pdf, init_pars,par_bounds)
    return pdf.expected_data(bestfit_nuisance_asimov)

##########################

def loglambdav(pars, data, pdf):
    return -2*pdf.logpdf(pars, data)

### The Test Statistic
def qmu(mu,data, pdf, init_pars,par_bounds):
    mubhathat = constrained_bestfit(mu,data,pdf, init_pars,par_bounds)
    muhatbhat = unconstrained_bestfit(data,pdf, init_pars,par_bounds)
    qmu = loglambdav(mubhathat, data,pdf)-loglambdav(muhatbhat, data,pdf)
    if muhatbhat[pdf.config.poi_index] > mu:
        return 0.0
    if -1e-6 < qmu <0:
        log.warning('WARNING: qmu negative: %s', qmu)
        return 0.0
    return qmu

### The Global Fit
def unconstrained_bestfit(data,pdf, init_pars,par_bounds):
    result = minimize(loglambdav, init_pars, method='SLSQP', args = (data,pdf), bounds = par_bounds)
    try:
        assert result.success
    except AssertionError:
        log.error(result)
    return result.x

### The Fit Conditions on a specific POI value
def constrained_bestfit(constrained_mu,data,pdf, init_pars,par_bounds):
    cons = {'type': 'eq', 'fun': lambda v: v[pdf.config.poi_index]-constrained_mu}
    result = minimize(loglambdav, init_pars, constraints=cons, method='SLSQP',args = (data,pdf), bounds = par_bounds)
    try:
        assert result.success
    except AssertionError:
        log.error(result)
    return result.x

def pvals_from_teststat(sqrtqmu_v,sqrtqmuA_v):
    CLsb = 1-norm.cdf(sqrtqmu_v )
    CLb  = norm.cdf(sqrtqmuA_v-sqrtqmu_v)
    CLs  = CLb/CLsb
    return CLsb,CLb,CLs

def runOnePoint(muTest, data,pdf, init_pars,par_bounds):
    asimov_mu = 0.0
    asimov_data = generate_asimov_data(asimov_mu,data,pdf,init_pars,par_bounds)

    # print 'ASIMOV MY FRIEND', asimov_data
    qmu_v  = qmu(muTest,data,pdf, init_pars,par_bounds)
    qmuA_v = qmu(muTest,asimov_data,pdf,init_pars,par_bounds)


    # print 'QMUS! MY FRIEND', qmu_v, qmuA_v

    sqrtqmu_v = np.sqrt(qmu_v)
    sqrtqmuA_v = np.sqrt(qmuA_v)

    sigma = muTest/sqrtqmuA_v if sqrtqmuA_v > 0 else None

    CLsb,CLb,CLs = pvals_from_teststat(sqrtqmu_v, sqrtqmuA_v)

    CLs_exp = []
    for nsigma in [-2,-1,0,1,2]:
        sqrtqmu_v_sigma =  sqrtqmuA_v-nsigma
        CLs_exp.append(pvals_from_teststat(sqrtqmu_v_sigma,sqrtqmuA_v)[-1])
    return qmu_v,qmuA_v,sigma,CLsb,CLb,CLs,CLs_exp
