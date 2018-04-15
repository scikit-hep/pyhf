import logging
import pyhf.optimize as optimize
import pyhf.tensor as tensor


log = logging.getLogger(__name__)
tensorlib = tensor.numpy_backend()
optimizer = optimize.scipy_optimizer()

def set_backend(backend):
    """
    Set the backend and the associated optimizer

    Args:
        backend: One of the supported pyhf backends: NumPy,
                 TensorFlow, PyTorch, and MXNet

    Returns:
        None

    Example:
        pyhf.set_backend(tensorflow_backend(session=tf.Session()))
    """
    global tensorlib
    global optimizer

    tensorlib = backend
    if isinstance(tensorlib, tensor.tensorflow_backend):
        optimizer = optimize.tflow_optimizer(tensorlib)
    elif isinstance(tensorlib,tensor.pytorch_backend):
        optimizer = optimize.pytorch_optimizer(tensorlib=tensorlib)
    # TODO: Add support for mxnet_optimizer()
    # elif isinstance(tensorlib, mxnet_backend):
    #     optimizer = mxnet_optimizer()
    else:
        optimizer = optimize.scipy_optimizer()

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

    def add_sample(self, channel, sample, modifier_data):
        self.at_minus_one.setdefault(channel['name'], {})[sample['name']] = modifier_data['lo']
        self.at_plus_one.setdefault(channel['name'], {})[sample['name']] = modifier_data['hi']

    def alphas(self, pars):
        return pars  # the nuisance parameters correspond directly to the alpha

    def expected_data(self, pars):
        return self.alphas(pars)

    def pdf(self, a, alpha):
        return tensorlib.normal(a, alpha, 1)

class histosys_constraint(object):

    def __init__(self):
        self.at_zero = {}
        self.at_minus_one = {}
        self.at_plus_one = {}
        self.auxdata = [0]  # observed data is always at a = 1

    def add_sample(self, channel, sample, modifier_data):
        self.at_zero.setdefault(channel['name'], {})[sample['name']] = sample['data']
        self.at_minus_one.setdefault(channel['name'], {})[sample['name']] = modifier_data['lo_data']
        self.at_plus_one.setdefault(channel['name'], {})[sample['name']] = modifier_data['hi_data']

    def alphas(self, pars):
        return pars  # the nuisance parameters correspond directly to the alpha

    def expected_data(self, pars):
        return self.alphas(pars)

    def pdf(self, a, alpha):
        return tensorlib.normal(a, alpha, [1])


class shapesys_constraint(object):

    def __init__(self, nom_data, modifier_data):
        self.auxdata = []
        self.bkg_over_db_squared = []
        for b, deltab in zip(nom_data, modifier_data):
            bkg_over_bsq = b * b / deltab / deltab  # tau*b
            log.info('shapesys for b,delta b (%s, %s) -> tau*b = %s',
                     b, deltab, bkg_over_bsq)
            self.bkg_over_db_squared.append(bkg_over_bsq)
            self.auxdata.append(bkg_over_bsq)

    def alphas(self, pars):
        return tensorlib.product(tensorlib.stack([pars, tensorlib.astensor(self.bkg_over_db_squared)]), axis=0)

    def pdf(self, a, alpha):
        return tensorlib.poisson(a, alpha)

    def expected_data(self, pars):
        return self.alphas(pars)

class modelconfig(object):
    @classmethod
    def from_spec(cls,spec,poiname = 'mu'):
        # hacky, need to keep track in which order we added the constraints
        # so that we can generate correctly-ordered data
        instance = cls()
        for channel in spec['channels']:
            for sample in channel['samples']:
                for modifier_def in sample['modifiers']:
                    instance.add_modifier_from_def(channel, sample, modifier_def)
        instance.set_poi(poiname)
        return instance

    def __init__(self):
        self.poi_index = None
        self.par_map = {}
        self.par_order = []
        self.auxdata = []
        self.auxdata_order = []
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

    def modifier(self, name):
        return self.par_map[name]['modifier']

    def set_poi(self,name):
        s = self.par_slice(name)
        assert s.stop-s.start == 1
        self.poi_index = s.start

    def add_modifier(self, name, npars, modifier, suggested_init, suggested_bounds):
        is_constraint = type(modifier) in [histosys_constraint, normsys_constraint, shapesys_constraint]
        if name in self.par_map:
            if type(modifier) == normsys_constraint:
                log.info('accepting existing normsys')
                return False
            if type(modifier) == histosys_constraint:
                log.info('accepting existing histosys')
                return False
            if type(modifier) == type(None):
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
            'modifier': modifier,
            'suggested_init': suggested_init,
            'suggested_bounds': suggested_bounds
        }
        if is_constraint:
            self.auxdata += self.modifier(name).auxdata
            self.auxdata_order.append(name)
        return True

    def add_modifier_from_def(self, channel, sample, modifier_def):
        if modifier_def['type'] == 'normfactor':
            modifier = None  # no object for factors
            self.add_modifier(name=modifier_def['name'],
                                modifier=modifier,
                                npars=1,
                                suggested_init=[1.0],
                                suggested_bounds=[[0, 10]])
        if modifier_def['type'] == 'shapefactor':
            modifier = None  # no object for factors
            self.add_modifier(name=modifier_def['name'],
                                modifier=modifier,
                                npars=len(sample['data']),
                                suggested_init   =[1.0] * len(sample['data']),
                                suggested_bounds=[[0, 10]] * len(sample['data'])
                        )
        if modifier_def['type'] == 'shapesys':
            # we reserve one parameter for each bin
            modifier = shapesys_constraint(sample['data'], modifier_def['data'])
            self.add_modifier(
                name=modifier_def['name'],
                npars=len(sample['data']),
                suggested_init=[1.0] * len(sample['data']),
                suggested_bounds=[[0, 10]] * len(sample['data']),
                modifier=modifier,
            )
        if modifier_def['type'] == 'normsys':
            modifier = normsys_constraint()
            self.add_modifier(name=modifier_def['name'],
                         npars=1,
                         modifier=modifier,
                         suggested_init=[0.0],
                         suggested_bounds=[[-5, 5]])
            self.modifier(modifier_def['name']).add_sample(channel, sample, modifier_def['data'])
        if modifier_def['type'] == 'histosys':
            modifier = histosys_constraint()
            self.add_modifier(
                modifier_def['name'],
                npars=1,
                modifier=modifier,
                suggested_init=[1.0],
                suggested_bounds=[[-5, 5]])
            self.modifier(modifier_def['name']).add_sample(channel, sample, modifier_def['data'])

class hfpdf(object):
    def __init__(self, spec, **config_kwargs):
        self.config = modelconfig.from_spec(spec,**config_kwargs)
        self.spec = spec

    def _multiplicative_factors(self, channel, sample, pars):
        multiplicative_types = ['shapesys', 'normfactor', 'shapefactor']
        modifiers = [m['name'] for m in sample['modifiers'] if m['type'] in multiplicative_types]
        return [pars[self.config.par_slice(m)] for m in modifiers]

    def _normsysfactor(self, channel, sample, pars):
        # normsysfactor(nom_sys_alphas)   = 1 + sum(interp(1, anchors[i][0],
        # anchors[i][0], val=alpha)  for i in range(nom_sys_alphas))
        modifiers = [m['name'] for m in sample['modifiers'] if m['type'] == 'normsys']
        factors = []
        for m in modifiers:
            modifier, modpars = self.config.modifier(m), pars[self.config.par_slice(m)]
            assert int(modpars.shape[0]) == 1
            mod_factor = _hfinterp_code1(modifier.at_minus_one[channel['name']][sample['name']],
                                         modifier.at_zero,
                                         modifier.at_plus_one[channel['name']][sample['name']],
                                         modpars)[0]
            factors.append(mod_factor)
        return tensorlib.product(factors)

    def _histosysdelta(self, channel, sample, pars):
        modifiers = [m['name'] for m in sample['modifiers']
                if m['type'] == 'histosys']
        stack = None
        for m in modifiers:
            modifier, modpars = self.config.modifier(m), pars[self.config.par_slice(m)]
            assert int(modpars.shape[0]) == 1

            # print 'MODPARS', type(modpars.data)

            mod_delta = _hfinterp_code0(modifier.at_minus_one[channel['name']][sample['name']],
                                        modifier.at_zero[channel['name']][sample['name']],
                                        modifier.at_plus_one[channel['name']][sample['name']],
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
        nom = tensorlib.astensor(sample['data'])
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
            thisaux = self.config.modifier(modname).expected_data(
                pars[self.config.par_slice(modname)])
            tocat = [thisaux] if auxdata is None else [auxdata, thisaux]
            auxdata = tensorlib.concatenate(tocat)
        return auxdata

    def expected_actualdata(self, pars):
        pars = tensorlib.astensor(pars)
        data = []
        for channel in self.spec['channels']:
            data.append(tensorlib.sum(tensorlib.stack([self.expected_sample(channel, sample, pars) for sample in channel['samples']]),axis=0))
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
            modifier, modslice = self.config.modifier(cname), \
                self.config.par_slice(cname)
            modalphas = modifier.alphas(pars[modslice])
            end_index = start_index + int(modalphas.shape[0])
            thisauxdata = auxdata[start_index:end_index]
            start_index = end_index
            constraint_term = tensorlib.log(modifier.pdf(thisauxdata, modalphas))
            summands = constraint_term if summands is None else tensorlib.concatenate([summands,constraint_term])
        return tensorlib.sum(summands) if summands is not None else 0

    def logpdf(self, pars, data):
        pars, data = tensorlib.astensor(pars), tensorlib.astensor(data)
        cut = int(data.shape[0]) - len(self.config.auxdata)
        actual_data, aux_data = data[:cut], data[cut:]
        lambdas_data = self.expected_actualdata(pars)
        summands = tensorlib.log(tensorlib.poisson(actual_data, lambdas_data))

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
