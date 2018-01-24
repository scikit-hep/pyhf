import numpy as np
from scipy.stats import poisson, norm
from scipy.optimize import minimize

def poisson_val(n,lam):
    #use continuous gaussian approx, b/c asimov data may not be integer
    return norm.pdf(n, loc = lam, scale = np.sqrt(lam))

def model_expected_background(nuisance_pars, constants):
    background_counts = nuisance_pars
    return background_counts

def model_expected_actualdata(s,poi,nuisance_pars, constants):
    signal_counts     = s(poi)
    background_counts = model_expected_background(nuisance_pars,constants)
    return [s+b for s,b in zip(signal_counts, background_counts)]

def model_expected_auxdata(nuisance_pars, constants):
    ### probably more correctly this should be the mode of the constraint_pdf
    background_counts = model_expected_background(nuisance_pars,constants)
    tau_constants     = constants
    return [tau*b for b,tau in zip(background_counts, tau_constants)]

def constraint_pars(nuisance_pars, constants):
    #convert nuisance parameters into form that is understood by
    #constraint_pdf (i.e alphas)
    return model_expected_auxdata(nuisance_pars, constants)

def constraint_pdf(auxdata, nuisance_pars,constants):
    #constraint_pdf
    constraint_alphas = constraint_pars(nuisance_pars, constants)
    constraint_func = poisson_val
    product = 1.0
    for a,alpha in zip(auxdata, constraint_alphas):
        product = product * constraint_func(a, alpha)
    return product

def model_pdf(pars, data, s, constants):
    poi = pars[0]
    nuisance_pars = pars[1:]

    lambdas_data = model_expected_actualdata(s,poi, nuisance_pars, constants)
    actual_data,aux_data = data[:len(lambdas_data)], data[len(lambdas_data):]

    product = 1
    for d,lam in zip(actual_data, lambdas_data):
        product = product * poisson_val(d, lam)

    product = product * constraint_pdf(aux_data, nuisance_pars,constants)
    return product

def model_expected_data(pars,s,constants):
    poi = pars[0]
    nuisance_pars = pars[1:]

    expected_actual      = model_expected_actualdata(s,poi, nuisance_pars, constants)
    expected_constraints = model_expected_auxdata(nuisance_pars, constants)
    expected = expected_actual + expected_constraints
    return expected

def generate_asimov_data(asimov_mu,data,constants,s, init_pars,par_bounds):
    bestfit_nuisance_asimov = constrained_bestfit(asimov_mu,data,constants,s, init_pars,par_bounds)
    return model_expected_data(bestfit_nuisance_asimov,s,constants)

##########################

def loglambdav(pars, data, constants, s):
    poi = pars[0]
    nuisance_pars = pars[1:]
    return -2*np.log(model_pdf(pars, data, s, constants))


### The Test Statistic
def qmu(mu,data,constants,s, init_pars,par_bounds):
    mubhathat = constrained_bestfit(mu,data,constants,s, init_pars,par_bounds)
    muhatbhat = unconstrained_bestfit(data,constants,s, init_pars,par_bounds)
    qmu = loglambdav(mubhathat, data, constants,s)-loglambdav(muhatbhat, data, constants,s)
    if muhatbhat[0] > mu:
        return 0.0
    if -1e-6 < qmu <0:
        print 'WARNING: qmu negative: ', qmu
        return 0.0
    return qmu

### The Global Fit
def unconstrained_bestfit(data, constants,s, init_pars,par_bounds):
    result = minimize(loglambdav, init_pars, method='SLSQP', args = (data,constants,s), bounds = par_bounds)
    try:
        assert result.success
    except AssertionError:
        print result
    return result.x

### The Fit Conditions on a specific POI value
def constrained_bestfit(constrained_mu,data, constants,s, init_pars,par_bounds):
    cons = {'type': 'eq', 'fun': lambda v: v[0]-constrained_mu}
    result = minimize(loglambdav, init_pars, constraints=cons, method='SLSQP',args = (data,constants,s), bounds = par_bounds)
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

def runOnePoint(muTest, data,constants,s,init_pars,par_bounds):
#     print 'DATALEN',len(data)
#     print 'CONSTANTLEN',len(constants)
#     print 'PARLEN',len(init_pars)

    asimov_mu = 0.0
    asimov_data = generate_asimov_data(asimov_mu,data,constants,s,init_pars,par_bounds)

    qmu_v  = qmu(muTest,data, constants,s, init_pars,par_bounds)
    qmuA_v = qmu(muTest,asimov_data, constants,s,init_pars,par_bounds)
    sqrtqmu_v = np.sqrt(qmu_v)
    sqrtqmuA_v = np.sqrt(qmuA_v)

    sigma = muTest/sqrtqmuA_v if sqrtqmuA_v > 0 else None

    CLsb,CLb,CLs = pvals_from_teststat(sqrtqmu_v, sqrtqmuA_v)

    CLs_exp = []
    for nsigma in [-2,-1,0,1,2]:
        sqrtqmu_v_sigma =  sqrtqmuA_v-nsigma
        CLs_exp.append(pvals_from_teststat(sqrtqmu_v_sigma,sqrtqmuA_v)[-1])
    return qmu_v,qmuA_v,sigma,CLsb,CLb,CLs,CLs_exp
