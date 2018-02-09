import logging

from .tensor.numpy_backend import numpy_backend
from .optimize.opt_scipy import scipy_optimizer
try:
    from .tensor.pytorch_backend import pytorch_backend
    from .optimize.opt_pytorch import pytorch_optimizer
    assert pytorch_backend
    assert pytorch_optimizer
except ImportError:
    pass

try:
    from .tensor.tensorflow_backend import tensorflow_backend
    from .optimize.opt_tflow import tflow_optimizer
    assert tensorflow_backend
    assert tflow_optimizer
except ImportError:
    pass


tensorlib = numpy_backend()
optimizer = scipy_optimizer()

log = logging.getLogger(__name__)

def _poisson_impl(n, lam):
    return tensorlib.poisson(n,lam)

def _gaussian_impl(x, mu, sigma):
    return tensorlib.normal(x,mu,sigma)

def _hfinterp_code0(at_minus_one, at_zero, at_plus_one, alphas):
    at_minus_one = tensorlib.astensor(at_minus_one)
    at_zero = tensorlib.astensor(at_zero)
    at_plus_one = tensorlib.astensor(at_plus_one)

    alphas = tensorlib.astensor(alphas)

    iplus_izero  = at_plus_one - at_zero
    izero_iminus = at_zero - at_minus_one

    mask = tensorlib.outer(alphas < 0, tensorlib.ones(iplus_izero.shape))
    return tensorlib.where(mask, tensorlib.outer(alphas, izero_iminus), tensorlib.outer(alphas, iplus_izero))

def _hfinterp_code1(at_minus_one, at_zero, at_plus_one, alphas):
    at_minus_one = tensorlib.astensor(at_minus_one)
    at_zero = tensorlib.astensor(at_zero)
    at_plus_one = tensorlib.astensor(at_plus_one)
    alphas = tensorlib.astensor(alphas)

    base_positive = tensorlib.divide(at_plus_one,  at_zero)
    base_negative = tensorlib.divide(at_minus_one, at_zero)
    expo_positive = tensorlib.outer(alphas, tensorlib.ones(base_positive.shape))
    mask = tensorlib.outer(alphas > 0, tensorlib.ones(base_positive.shape))
    bases = tensorlib.where(mask,base_positive,base_negative)
    exponents = tensorlib.where(mask, expo_positive,-expo_positive)
    return tensorlib.power(bases, exponents)

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
        return _gaussian_impl(a, alpha, [1])


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
        return tensorlib.product(tensorlib.stack([pars, tensorlib.astensor(self.bkg_over_db_squared)]), axis=0)

    def pdf(self, a, alpha):
        return _poisson_impl(a, alpha)

    def expected_data(self, pars):
        return self.alphas(pars)

class modelconfig(object):
    @classmethod
    def from_spec(cls,spec,poiname = 'mu'):
        # hacky, need to keep track in which order we added the constraints
        # so that we can generate correctly-ordered data
        instance = cls()
        for ch, samples in spec.items():
            instance.channel_order.append(ch)
            for sample, sample_def in samples.items():
                for mod_def in sample_def['mods']:
                    instance.add_mod_from_def(ch, sample, sample_def, mod_def)
        instance.set_poi(poiname)
        return instance

    def __init__(self):
        self.poi_index = None
        self.par_map = {}
        self.par_order = []
        self.auxdata = []
        self.auxdata_order = []
        self.channel_order = []
        self.next_index = 0

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

    def set_poi(self,name):
        s = self.par_slice(name)
        assert s.stop-s.start == 1
        self.poi_index = s.start

    def add_mod(self, name, npars, mod, suggested_init, suggested_bounds):
        is_constraint = type(mod) in [histosys_constraint, normsys_constraint, shapesys_constraint]
        if name in self.par_map:
            if type(mod) == normsys_constraint:
                log.info('accepting existing normsys')
                return False
            if type(mod) == histosys_constraint:
                log.info('accepting existing histosys')
                return False
            if type(mod) == type(None):
                log.info('accepting existing unconstrained factor ')
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
        if is_constraint:
            self.auxdata += self.mod(name).auxdata
            self.auxdata_order.append(name)
        return True

    def add_mod_from_def(self, ch, sample, sample_def, mod_def):
        if mod_def['type'] == 'normfactor':
            mod = None  # no object for factors
            self.add_mod(name=mod_def['name'],
                                mod=mod,
                                npars=1,
                                suggested_init=[1.0],
                                suggested_bounds=[[0, 10]])
        if mod_def['type'] == 'shapefactor':
            mod = None  # no object for factors
            self.add_mod(name=mod_def['name'],
                                mod=mod,
                                npars=len(sample_def['data']),
                                suggested_init   =[1.0] * len(sample_def['data']),
                                suggested_bounds=[[0, 10]] * len(sample_def['data'])
                        )
        if mod_def['type'] == 'shapesys':
            # we reserve one parameter for each bin
            mod = shapesys_constraint(sample_def['data'], mod_def['data'])
            self.add_mod(
                name=mod_def['name'],
                npars=len(sample_def['data']),
                suggested_init=[1.0] * len(sample_def['data']),
                suggested_bounds=[[0, 10]] * len(sample_def['data']),
                mod=mod,
            )
        if mod_def['type'] == 'normsys':
            mod = normsys_constraint()
            self.add_mod(name=mod_def['name'],
                         npars=1,
                         mod=mod,
                         suggested_init=[0.0],
                         suggested_bounds=[[-5, 5]])
            self.mod(mod_def['name']).add_sample(
                ch, sample, sample_def['data'], mod_def['data']
            )
        if mod_def['type'] == 'histosys':
            mod = histosys_constraint()
            self.add_mod(
                mod_def['name'],
                npars=1,
                mod=mod,
                suggested_init=[1.0],
                suggested_bounds=[[-5, 5]])
            self.mod(mod_def['name']).add_sample(
                ch, sample, sample_def['data'], mod_def['data']
            )

class hfpdf(object):
    def __init__(self, spec, **config_kwargs):
        self.config = modelconfig.from_spec(spec,**config_kwargs)
        self.channels = spec

    def _multiplicative_factors(self, channel, sample, pars):
        multiplicative_types = ['shapesys', 'normfactor', 'shapefactor']
        mods = [m['name'] for m in self.channels[channel][sample]['mods']
                if m['type'] in multiplicative_types]
        return [pars[self.config.par_slice(m)] for m in mods]

    def _normsysfactor(self, channel, sample, pars):
        # normsysfactor(nom_sys_alphas)   = 1 + sum(interp(1, anchors[i][0],
        # anchors[i][0], val=alpha)  for i in range(nom_sys_alphas))
        mods = [m['name'] for m in self.channels[channel][sample]['mods']
                if m['type'] == 'normsys']
        factors = []
        for m in mods:
            mod, modpars = self.config.mod(m), pars[self.config.par_slice(m)]
            assert int(modpars.shape[0]) == 1
            mod_factor = _hfinterp_code1(mod.at_minus_one[channel][sample],
                                         mod.at_zero,
                                         mod.at_plus_one[channel][sample],
                                         modpars)[0]
            factors.append(mod_factor)
        return tensorlib.product(factors)

    def _histosysdelta(self, channel, sample, pars):
        mods = [m['name'] for m in self.channels[channel][sample]['mods']
                if m['type'] == 'histosys']
        stack = None
        for m in mods:
            mod, modpars = self.config.mod(m), pars[self.config.par_slice(m)]
            assert int(modpars.shape[0]) == 1

            # print 'MODPARS', type(modpars.data)

            mod_delta = _hfinterp_code0(mod.at_minus_one[channel][sample],
                                         mod.at_zero[channel][sample],
                                         mod.at_plus_one[channel][sample],
                                         modpars)[0]
            stack = tensorlib.stack([mod_delta]) if stack is None else tensorlib.stack([stack,mod_delta])

        return tensorlib.sum(stack, axis=0) if stack is not None else None

    def expected_sample(self, channel, sample, pars):
        # for each sample the expected ocunts are
        # counts = (multiplicative factors) * (normsys multiplier) * (histsys delta + nominal hist)
        #        = f1*f2*f3*f4* nomsysfactor(nom_sys_alphas) * hist(hist_addition(histosys_alphas) + nomdata)
        # nomsysfactor(nom_sys_alphas)   = 1 + sum(interp(1, anchors[i][0], anchors[i][0], val=alpha)  for i in range(nom_sys_alphas))
        # hist_addition(histosys_alphas) = sum(interp(nombin, anchors[i][0],
        # anchors[i][0], val=alpha) for i in range(histosys_alphas))
        nom = tensorlib.astensor(self.channels[channel][sample]['data'])
        histosys_delta = self._histosysdelta(channel, sample, pars)

        interp_histo = tensorlib.sum(tensorlib.stack([nom, histosys_delta]), axis=0) if (histosys_delta is not None) else nom

        factors = []
        factors += self._multiplicative_factors(channel, sample, pars)
        factors += [self._normsysfactor(channel, sample, pars)]
        factors += [interp_histo]
        return tensorlib.product(tensorlib.stack(tensorlib.simple_broadcast(*factors)), axis=0)

    def expected_auxdata(self, pars):
        # probably more correctly this should be the expectation value of the constraint_pdf
        # or for the constraints we are using (single par constraings with mean == mode), we can
        # just return the alphas

        # order matters! because we generated auxdata in a certain order
        auxdata = None
        for modname in self.config.auxdata_order:
            thisaux = self.config.mod(modname).expected_data(
                pars[self.config.par_slice(modname)])
            tocat = [thisaux] if auxdata is None else [auxdata, thisaux]
            auxdata = tensorlib.concatenate(tocat)
        return auxdata

    def expected_actualdata(self, pars):
        pars = tensorlib.astensor(pars)
        data = []
        for channel in self.config.channel_order:
            data.append(tensorlib.sum(tensorlib.stack([self.expected_sample(channel, sample, pars) for sample in self.channels[channel]]),axis=0))
        return tensorlib.concatenate(data)

    def expected_data(self, pars, include_auxdata=True):
        pars = tensorlib.astensor(pars)
        expected_actual = self.expected_actualdata(pars)

        if not include_auxdata:
            return expected_actual
        expected_constraints = self.expected_auxdata(pars)
        tocat = [expected_actual] if expected_constraints is None else [expected_actual,expected_constraints]
        return tensorlib.concatenate(tocat)

    def constraint_logpdf(self, auxdata, pars):
        # iterate over all constraints order doesn't matter....
        start_index = 0
        summands = None
        for cname in self.config.auxdata_order:
            mod, modslice = self.config.mod(cname), \
                self.config.par_slice(cname)
            modalphas = mod.alphas(pars[modslice])
            end_index = start_index + int(modalphas.shape[0])
            thisauxdata = auxdata[start_index:end_index]
            start_index = end_index
            constraint_term = tensorlib.log(mod.pdf(thisauxdata, modalphas))
            summands = constraint_term if summands is None else tensorlib.concatenate([summands,constraint_term])
        return tensorlib.sum(summands) if summands is not None else 0

    def logpdf(self, pars, data):
        pars, data = tensorlib.astensor(pars), tensorlib.astensor(data)
        cut = int(data.shape[0]) - len(self.config.auxdata)
        actual_data, aux_data = data[:cut], data[cut:]
        lambdas_data = self.expected_actualdata(pars)
        summands = tensorlib.log(_poisson_impl(actual_data, lambdas_data))

        result = tensorlib.sum(summands) + self.constraint_logpdf(aux_data, pars)
        return tensorlib.astensor(result) * tensorlib.ones((1)) #ensure (1,) array shape also for numpy

    def pdf(self, pars, data):
        return tensorlib.exp(self.logpdf(pars, data))


def generate_asimov_data(asimov_mu, data, pdf, init_pars, par_bounds):
    bestfit_nuisance_asimov = optimizer.constrained_bestfit(
        loglambdav, asimov_mu, data, pdf, init_pars, par_bounds)
    return pdf.expected_data(bestfit_nuisance_asimov)

##########################


def loglambdav(pars, data, pdf):
    return -2 * pdf.logpdf(pars, data)

def qmu(mu, data, pdf, init_pars, par_bounds):
    # The Test Statistic
    mubhathat = tensorlib.tolist(optimizer.constrained_bestfit(loglambdav, mu, data, pdf, init_pars, par_bounds))
    muhatbhat = tensorlib.tolist(optimizer.unconstrained_bestfit(loglambdav, data, pdf, init_pars, par_bounds))
    qmu = tensorlib.tolist(loglambdav(mubhathat, data, pdf) - loglambdav(muhatbhat, data, pdf))[0]
    if muhatbhat[pdf.config.poi_index] > mu:
        return 0.0
    if -1e-6 < qmu < 0:
        log.warning('WARNING: qmu negative: %s', qmu)
        return 0.0
    return qmu

from scipy.stats import norm
def pvals_from_teststat(sqrtqmu_v, sqrtqmuA_v):
    CLsb = 1 - norm.cdf(sqrtqmu_v)
    CLb = norm.cdf(sqrtqmuA_v - sqrtqmu_v)
    CLs = CLb / CLsb
    return CLsb, CLb, CLs

import math
def runOnePoint(muTest, data, pdf, init_pars, par_bounds):
    asimov_mu = 0.0
    asimov_data = tensorlib.tolist(generate_asimov_data(asimov_mu, data,
                                   pdf, init_pars, par_bounds))

    qmu_v  = qmu(muTest, data, pdf, init_pars, par_bounds)
    qmuA_v = qmu(muTest, asimov_data, pdf, init_pars, par_bounds)

    sqrtqmu_v  = math.sqrt(qmu_v)
    sqrtqmuA_v = math.sqrt(qmuA_v)

    sigma = muTest / sqrtqmuA_v if sqrtqmuA_v > 0 else None

    CLsb, CLb, CLs = pvals_from_teststat(sqrtqmu_v, sqrtqmuA_v)

    CLs_exp = []
    for nsigma in [-2, -1, 0, 1, 2]:
        sqrtqmu_v_sigma = sqrtqmuA_v - nsigma
        CLs_exp.append(pvals_from_teststat(sqrtqmu_v_sigma, sqrtqmuA_v)[-1])
    return qmu_v, qmuA_v, sigma, CLsb, CLb, CLs, CLs_exp
