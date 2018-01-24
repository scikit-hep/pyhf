import numpy as np
from scipy.stats import poisson, norm
from scipy.optimize import minimize

class hfpdf(object):
    def __init__(self, signal_data, bkg_data, bkg_uncerts):
        self.samples = {
            'signal': {
                'data': signal_data
            },
            'background': {
                'data': bkg_data,
                'systs': [
                    {
                        'type': 'shapesys',
                        'data': bkg_uncerts
                    }
                ]
            }
        }
        self.auxdata   = []
        self.background_taus = []
        for b, deltab in zip(
                self.samples['background']['data'],
                self.samples['background']['systs'][0]['data']):
            tau = b/deltab/deltab
            self.background_taus.append(tau)
            self.auxdata.append(tau*b)

    def expected_signal(self, poi, nuisance_pars):
        nominal_signals = self.samples['signal']['data']
        return [poi*snom for snom in nominal_signals]

    def expected_background(self,nuisance_pars):
        background_counts = nuisance_pars
        return background_counts

    def expected_auxdata(self, nuisance_pars):
        ### probably more correctly this should be the expectation value of the constraint_pdf
        ### or for the constraints we are using (with mean == mode), the mode
        ### could be taken from a pdf object via pdf.expectation_value(pars)
        background_counts = self.expected_background(nuisance_pars)
        return [tau*b for b,tau in zip(background_counts, self.background_taus)]

    def expected_actualdata(self, poi, nuisance_pars):
        signal_counts     = self.expected_signal(poi, nuisance_pars)
        background_counts = self.expected_background(nuisance_pars)
        return [s+b for s,b in zip(signal_counts, background_counts)]

    def expected_data(self, pars, include_auxdata = True):
        poi = pars[0]
        nuisance_pars = pars[1:]

        expected_actual = self.expected_actualdata(poi, nuisance_pars)
        if not include_auxdata:
            return expected_actual

        expected_constraints = self.expected_auxdata(nuisance_pars)
        return expected_actual + expected_constraints

    @staticmethod
    def _poisson_impl(n, lam):
        #use continuous gaussian approx, b/c asimov data may not be integer
        return norm.pdf(n, loc = lam, scale = np.sqrt(lam))

    def _constraint_pars(self, nuisance_pars):
        #convert nuisance parameters into form that is understood by
        #constraint_pdf (i.e alphas)
        return self.expected_auxdata(nuisance_pars)

    def constraint_pdf(self,auxdata, nuisance_pars):
        #constraint_pdf
        constraint_alphas = self._constraint_pars(nuisance_pars)
        constraint_func = self._poisson_impl
        product = 1.0
        for a,alpha in zip(auxdata, constraint_alphas):
            product = product * constraint_func(a, alpha)
        return product

    def pdf(self, pars, data):
        poi = pars[0]
        nuisance_pars = pars[1:]

        lambdas_data = self.expected_actualdata(poi, nuisance_pars)

        # split data in actual data and aux_data (inputs into constraints)
        actual_data,aux_data = data[:len(lambdas_data)], data[len(lambdas_data):]

        product = 1
        for d,lam in zip(actual_data, lambdas_data):
            product = product * self._poisson_impl(d, lam)

        product = product * self.constraint_pdf(aux_data, nuisance_pars)
        return product


def generate_asimov_data(asimov_mu, data, pdf, init_pars,par_bounds):
    bestfit_nuisance_asimov = constrained_bestfit(asimov_mu,data, pdf, init_pars,par_bounds)
    return pdf.expected_data(bestfit_nuisance_asimov)

##########################

def loglambdav(pars, data, pdf):
    poi = pars[0]
    nuisance_pars = pars[1:]
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

    # print 'asimov!', asimov_data

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
