import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import logging


log = logging.getLogger(__name__)


def _poisson_impl(n, lam):
    # use continuous gaussian approx, b/c asimov data may not be integer
    # print 'pois', n,lam
    # return poisson.pmf(n,lam)
    # print 'lam,sqrtlam',lam,np.sqrt(lam)
    # from scipy.special import gammaln
    # return np.exp(n*np.log(lam)-lam-gammaln(n+1.))
    return norm.pdf(n, loc=lam, scale=np.sqrt(lam))


def _gaussian_impl(x, mu, sigma):
    # use continuous gaussian approx, b/c asimov data may not be integer
    # print 'gaus', x, mu, sigma
    return norm.pdf(x, loc=mu, scale=sigma)


def _hfinterp_code0(at_minus_one, at_zero, at_plus_one):
    at_minus_one = np.array(at_minus_one)
    at_zero = np.array(at_zero)
    at_plus_one = np.array(at_plus_one)

    def func(alphas):
        alphas = np.array(alphas)
        iplus_izero = at_plus_one - at_zero
        izero_iminus = at_zero - at_minus_one
        mask = np.outer(alphas < 0, np.ones(iplus_izero.shape))
        interpolated = np.where(mask, np.outer(alphas, izero_iminus),
                                np.outer(alphas, iplus_izero))
        return interpolated
    return func


def _hfinterp_code1(at_minus_one, at_zero, at_plus_one):
    def func(alpha):
        alpha = np.array(alpha)
        base = np.where(alpha < 0, at_minus_one / at_zero,
                        at_plus_one / at_zero)
        exponents = np.where(alpha < 0, -alpha, alpha)
        return np.power(base, exponents)
    return func


class normsys_constraint(object):

    def __init__(self):
        self.at_zero = 1
        self.at_minus_one = {}
        self.at_plus_one = {}
        self.auxdata = [0]  # observed data is always at a = 1

    def add_sample(self, channel, sample, nominal, mod_data):
        self.at_minus_one.setdefault(channel, {})[sample] = mod_data['lo']
        self.at_plus_one.setdefault(channel, {})[sample] = mod_data['hi']

    def alphas(self, pars):
        return pars  # the nuisance parameters correspond directly to the alpha

    def expected_data(self, pars):
        return self.alphas(pars)

    def pdf(self, a, alpha):
        return _gaussian_impl(a, alpha, 1)


class histosys_constraint(object):

    def __init__(self):
        self.at_zero = {}
        self.at_minus_one = {}
        self.at_plus_one = {}
        self.auxdata = [0]  # observed data is always at a = 1

    def add_sample(self, channel, sample, nominal, mod_data):
        self.at_zero.setdefault(channel, {})[sample] = nominal
        self.at_minus_one.setdefault(channel, {})[sample] = mod_data['lo_hist']
        self.at_plus_one.setdefault(channel, {})[sample] = mod_data['hi_hist']

    def alphas(self, pars):
        return pars  # the nuisance parameters correspond directly to the alpha

    def expected_data(self, pars):
        return self.alphas(pars)

    def pdf(self, a, alpha):
        return _gaussian_impl(a, alpha, 1)


class shapesys_constraint(object):

    def __init__(self, nom_data, mod_data):
        self.auxdata = []
        self.bkg_over_db_squared = []
        for b, deltab in zip(nom_data, mod_data):
            bkg_over_bsq = b * b / deltab / deltab  # tau*b
            log.info('shapesys for b,delta b (%s, %s) -> tau*b = %s',
                     b, deltab, bkg_over_bsq)
            self.bkg_over_db_squared.append(bkg_over_bsq)
            self.auxdata.append(bkg_over_bsq)

    def alphas(self, pars):
        return np.product([pars, self.bkg_over_db_squared], axis=0)

    def pdf(self, a, alpha):
        return _poisson_impl(a, alpha)

    def expected_data(self, pars):
        return self.alphas(pars)


class modelconfig(object):

    def __init__(self):
        self.poi_index = None
        self.par_map = {}
        self.next_index = 0
        self.snapshots = {}
        self.par_order = []

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

    def par_slice(self, name):
        return self.par_map[name]['slice']

    def mod(self, name):
        return self.par_map[name]['mod']

    def add_mod(self, name, npars, mod, suggested_init, suggested_bounds):
        if name in self.par_map:
            if type(mod) == normsys_constraint:
                log.info('accepting existing normsys')
                return False
            if type(mod) == histosys_constraint:
                log.info('accepting existing histosys')
                return False
            raise RuntimeError(
                'shared systematic not implemented yet (processing {})'.format(name))
        log.info('adding modifier %s (%s new nuisance parameters)', name, npars)

        sl = slice(self.next_index, self.next_index + npars)
        self.next_index = self.next_index + npars
        self.par_order.append(name)
        self.par_map[name] = {
            'slice': sl,
            'mod': mod,
            'suggested_init': suggested_init,
            'suggested_bounds': suggested_bounds
        }
        return True


class hfpdf(object):

    def __init__(self, channels):
        self.channels = channels
        self.config = modelconfig()
        self.auxdata = []
        # hacky, need to keep track in which order we added the constraints
        # so that we can generate correctly-ordered data
        self.auxdata_order = []
        self.channel_order = []
        for ch, samples in self.channels.items():
            self.channel_order.append(ch)
            for sample, sample_def in samples.items():
                for mod_def in sample_def['mods']:
                    if mod_def['type'] == 'normfactor':
                        mod = None  # no object for factors
                        self.config.add_mod(name=mod_def['name'],
                                            mod=mod,
                                            npars=1,
                                            suggested_init=[1.0],
                                            suggested_bounds=[[0, 10]])
                        self.config.poi_index = self.config.par_slice(
                            mod_def['name']).start
                    if mod_def['type'] == 'shapesys':
                        # we reserve one parameter for each bin
                        mod = shapesys_constraint(sample_def['data'],
                                                  mod_def['data'])
                        just_added = self.config.add_mod(
                            name=mod_def['name'],
                            npars=len(sample_def['data']),
                            suggested_init=[1.0] * len(sample_def['data']),
                            suggested_bounds=[[0, 10]] *
                            len(sample_def['data']),
                            mod=mod,
                        )
                        if just_added:
                            self.auxdata += self.config.mod(
                                mod_def['name']).auxdata
                            self.auxdata_order.append(mod_def['name'])
                    if mod_def['type'] == 'normsys':
                        mod = normsys_constraint()
                        just_added = self.config.add_mod(name=mod_def['name'],
                                                         npars=1,
                                                         mod=mod,
                                                         suggested_init=[0.0],
                                                         suggested_bounds=[[-5, 5]])
                        self.config.mod(mod_def['name']).add_sample(
                            ch, sample, sample_def['data'], mod_def['data'])
                        if just_added:
                            self.auxdata += self.config.mod(
                                mod_def['name']).auxdata
                            self.auxdata_order.append(mod_def['name'])
                    if mod_def['type'] == 'histosys':
                        mod = histosys_constraint()
                        self.config.add_mod(
                            mod_def['name'],
                            npars=1,
                            mod=mod,
                            suggested_init=[1.0],
                            suggested_bounds=[[-5, 5]])
                        self.config.mod(mod_def['name']).add_sample(
                            ch, sample, sample_def['data'], mod_def['data']
                        )
                        self.auxdata += self.config.mod(
                            mod_def['name']).auxdata
                        self.auxdata_order.append(mod_def['name'])

    def _multiplicative_factors(self, channel, sample, pars):
        multiplicative_types = ['shapesys', 'normfactor']
        mods = [m['name'] for m in self.channels[channel][sample]['mods']
                if m['type'] in multiplicative_types]
        return [pars[self.config.par_slice(m)] for m in mods]

    def _normsysfactor(self, channel, sample, pars):
        # normsysfactor(nom_sys_alphas)   = 1 + sum(interp(1, anchors[i][0],
        # anchors[i][0], val=alpha)  for i in range(nom_sys_alphas))
        mods = [m['name'] for m in self.channels[channel][sample]['mods']
                if m['type'] == 'normsys']
        factor = 1

        for m in mods:
            mod = self.config.mod(m)
            val = pars[self.config.par_slice(m)]
            assert len(val) == 1
            interp = _hfinterp_code1(mod.at_minus_one[channel][sample],
                                     mod.at_zero, mod.at_plus_one[channel][sample])
            factor = factor * interp(val[0])
        return factor

    def _histosysdelta(self, channel, sample, pars):
        mods = [m['name'] for m in self.channels[channel][sample]['mods']
                if m['type'] == 'histosys']
        summands = []
        for m in mods:
            mod, modpars = self.config.mod(m), pars[self.config.par_slice(m)]
            assert len(modpars) == 1
            interpfunc = _hfinterp_code0(mod.at_minus_one[channel][sample],
                                         mod.at_zero[channel][sample],
                                         mod.at_plus_one[channel][sample])
            mod_delta = interpfunc(modpars)[0]
            summands.append(mod_delta)
        return np.sum(summands, axis=0)

    def expected_sample(self, channel, sample, pars):
        # for each sample the expected ocunts are
        # counts = (multiplicative factors) * (normsys multiplier) * (histsys delta + nominal hist)
        #        = f1*f2*f3*f4* nomsysfactor(nom_sys_alphas) * hist(hist_addition(histosys_alphas) + nomdata)
        # nomsysfactor(nom_sys_alphas)   = 1 + sum(interp(1, anchors[i][0], anchors[i][0], val=alpha)  for i in range(nom_sys_alphas))
        # hist_addition(histosys_alphas) = sum(interp(nombin, anchors[i][0],
        # anchors[i][0], val=alpha) for i in range(histosys_alphas))
        nom = self.channels[channel][sample]['data']
        histosys_delta = self._histosysdelta(channel, sample, pars)
        interp_histo = np.sum([nom, histosys_delta], axis=0)

        factors = []
        factors += self._multiplicative_factors(channel, sample, pars)
        factors += [self._normsysfactor(channel, sample, pars)]
        factors += [interp_histo]
        return np.prod(np.vstack(np.broadcast_arrays(*factors)), axis=0)

    def expected_auxdata(self, pars):
        # probably more correctly this should be the expectation value of the constraint_pdf
        # or for the constraints we are using (single par constraings with mean == mode), we can
        # just return the alphas

        # order matters! because we generated auxdata in a certain order
        auxdata = []
        for modname in self.auxdata_order:
            thisaux = self.config.mod(modname).expected_data(
                pars[self.config.par_slice(modname)])
            auxdata = np.append(auxdata, thisaux)
        return auxdata

    def expected_actualdata(self, pars):
        data = []
        for channel in self.channel_order:
            counts = [self.expected_sample(channel, sample, pars)
                      for sample in self.channels[channel]]
            data += [sum(sample_counts) for sample_counts in zip(*counts)]
        return data

    def expected_data(self, pars, include_auxdata=True):
        expected_actual = self.expected_actualdata(pars)

        if not include_auxdata:
            return expected_actual
        expected_constraints = self.expected_auxdata(pars)
        return np.concatenate([expected_actual, expected_constraints])

    def constraint_logpdf(self, auxdata, pars):
        # iterate over all constraints order doesn't matter....
        start_index = 0
        summands = []
        for cname in self.auxdata_order:
            mod, modslice = self.config.mod(cname), \
                self.config.par_slice(cname)
            modalphas = mod.alphas(pars[modslice])
            end_index = start_index + len(modalphas)
            thisauxdata = auxdata[start_index:end_index]
            start_index = end_index
            summands = np.concatenate([summands,
                                       np.log(mod.pdf(thisauxdata, modalphas))])
        return np.sum(summands)

    def logpdf(self, pars, data):
        cut = len(data) - len(self.auxdata)
        actual_data, aux_data = data[:cut], data[cut:]
        lambdas_data = self.expected_actualdata(pars)
        summands = np.log(_poisson_impl(actual_data, lambdas_data))
        return np.sum(summands) + self.constraint_logpdf(aux_data, pars)

    def pdf(self, pars, data):
        return np.exp(self.logpdf(pars, data))


def generate_asimov_data(asimov_mu, data, pdf, init_pars, par_bounds):
    bestfit_nuisance_asimov = constrained_bestfit(
        asimov_mu, data, pdf, init_pars, par_bounds)
    return pdf.expected_data(bestfit_nuisance_asimov)

##########################


def loglambdav(pars, data, pdf):
    return -2 * pdf.logpdf(pars, data)


def qmu(mu, data, pdf, init_pars, par_bounds):
    # The Test Statistic
    mubhathat = constrained_bestfit(mu, data, pdf, init_pars, par_bounds)
    muhatbhat = unconstrained_bestfit(data, pdf, init_pars, par_bounds)
    qmu = loglambdav(mubhathat, data, pdf) - loglambdav(muhatbhat, data, pdf)
    if muhatbhat[pdf.config.poi_index] > mu:
        return 0.0
    if -1e-6 < qmu < 0:
        log.warning('WARNING: qmu negative: %s', qmu)
        return 0.0
    return qmu


def unconstrained_bestfit(data, pdf, init_pars, par_bounds):
    # The Global Fit
    result = minimize(loglambdav, init_pars, method='SLSQP',
                      args=(data, pdf), bounds=par_bounds)
    try:
        assert result.success
    except AssertionError:
        log.error(result)
    return result.x


def constrained_bestfit(constrained_mu, data, pdf, init_pars, par_bounds):
    # The Fit Conditions on a specific POI value
    cons = {'type': 'eq', 'fun': lambda v: v[
        pdf.config.poi_index] - constrained_mu}
    result = minimize(loglambdav, init_pars, constraints=cons,
                      method='SLSQP', args=(data, pdf), bounds=par_bounds)
    try:
        assert result.success
    except AssertionError:
        log.error(result)
    return result.x


def pvals_from_teststat(sqrtqmu_v, sqrtqmuA_v):
    CLsb = 1 - norm.cdf(sqrtqmu_v)
    CLb = norm.cdf(sqrtqmuA_v - sqrtqmu_v)
    CLs = CLb / CLsb
    return CLsb, CLb, CLs


def runOnePoint(muTest, data, pdf, init_pars, par_bounds):
    asimov_mu = 0.0
    asimov_data = generate_asimov_data(asimov_mu, data,
                                       pdf, init_pars, par_bounds)

    qmu_v = qmu(muTest, data, pdf, init_pars, par_bounds)
    qmuA_v = qmu(muTest, asimov_data, pdf, init_pars, par_bounds)

    sqrtqmu_v = np.sqrt(qmu_v)
    sqrtqmuA_v = np.sqrt(qmuA_v)

    sigma = muTest / sqrtqmuA_v if sqrtqmuA_v > 0 else None

    CLsb, CLb, CLs = pvals_from_teststat(sqrtqmu_v, sqrtqmuA_v)

    CLs_exp = []
    for nsigma in [-2, -1, 0, 1, 2]:
        sqrtqmu_v_sigma = sqrtqmuA_v - nsigma
        CLs_exp.append(pvals_from_teststat(sqrtqmu_v_sigma, sqrtqmuA_v)[-1])
    return qmu_v, qmuA_v, sigma, CLsb, CLb, CLs, CLs_exp
