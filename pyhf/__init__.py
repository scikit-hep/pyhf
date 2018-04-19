import logging
import pyhf.optimize as optimize
import pyhf.tensor as tensor
from . import exceptions

log = logging.getLogger(__name__)
tensorlib = tensor.numpy_backend()
default_backend = tensorlib
optimizer = optimize.scipy_optimizer()
default_optimizer = optimizer

def get_backend():
    """
    Get the current backend and the associated optimizer

    Returns:
        backend, optimizer

    Example:
        backend, _ = pyhf.get_backend()
    """
    global tensorlib
    global optimizer
    return tensorlib, optimizer

# modifiers need access to tensorlib
# make sure import is below get_backend()
from . import modifiers


def set_backend(backend):
    """
    Set the backend and the associated optimizer

    Args:
        backend: One of the supported pyhf backends: NumPy,
                 TensorFlow, PyTorch, and MXNet

    Returns:
        None

    Example:
        import pyhf.tensor as tensor
        import tensorflow as tf
        pyhf.set_backend(tensor.tensorflow_backend(session=tf.Session()))
    """
    global tensorlib
    global optimizer

    tensorlib = backend
    if isinstance(tensorlib, tensor.tensorflow_backend):
        optimizer = optimize.tflow_optimizer(tensorlib)
    elif isinstance(tensorlib, tensor.pytorch_backend):
        optimizer = optimize.pytorch_optimizer(tensorlib=tensorlib)
    # TODO: Add support for mxnet_optimizer()
    # elif isinstance(tensorlib, tensor.mxnet_backend):
    #     optimizer = mxnet_optimizer()
    else:
        optimizer = optimize.scipy_optimizer()

class modelconfig(object):
    @classmethod
    def from_spec(cls,spec,poiname = 'mu'):
        # hacky, need to keep track in which order we added the constraints
        # so that we can generate correctly-ordered data
        instance = cls()
        for channel in spec['channels']:
            for sample in channel['samples']:
                for modifier_def in sample['modifiers']:
                    modifier = instance.add_or_get_modifier(channel, sample, modifier_def)
                    modifier.add_sample(channel, sample, modifier_def)
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
            init = init + self.par_map[name]['modifier'].suggested_init
        return init

    def suggested_bounds(self):
        bounds = []
        for name in self.par_order:
            bounds = bounds + self.par_map[name]['modifier'].suggested_bounds
        return bounds

    def par_slice(self, name):
        return self.par_map[name]['slice']

    def modifier(self, name):
        return self.par_map[name]['modifier']

    def set_poi(self,name):
        s = self.par_slice(name)
        assert s.stop-s.start == 1
        self.poi_index = s.start

    def add_or_get_modifier(self, channel, sample, modifier_def):
        """
        Add a new modifier if it does not exist and return it
        or get the existing modifier and return it

        Args:
            channel: current channel object (e.g. from spec)
            sample: current sample object (e.g. from spec)
            modifier_def: current modifier definitions (e.g. from spec)

        Returns:
            modifier object

        Example:
            modifier = instance.add_or_get_modifier(channel, sample, modifier_def)
        """
        # get modifier class associated with modifier type
        try:
            modifier_cls = modifiers.registry[modifier_def['type']]
        except KeyError:
            log.exception('Modifier type not implemented yet (processing {0:s}). Current modifier types: {1}'.format(modifier_def['type'], modifiers.registry.keys()))
            raise exceptions.InvalidModifier()

        # if modifier is shared, check if it already exists and use it
        if modifier_cls.is_shared and modifier_def['name'] in self.par_map:
            log.info('using existing shared, {0:s}constrained modifier (name={1:s}, type={2:s})'.format('' if modifier_cls.is_constrained else 'un', modifier_def['name'], modifier_cls.__name__))
            return self.par_map[modifier_def['name']]['modifier']

        # did not return, so create new modifier and return it
        modifier = modifier_cls(sample['data'], modifier_def['data'])
        npars = modifier.n_parameters

        log.info('adding modifier %s (%s new nuisance parameters)', modifier_def['name'], npars)
        sl = slice(self.next_index, self.next_index + npars)
        self.next_index = self.next_index + npars
        self.par_order.append(modifier_def['name'])
        self.par_map[modifier_def['name']] = {
            'slice': sl,
            'modifier': modifier
        }
        if modifier.is_constrained:
            self.auxdata += self.modifier(modifier_def['name']).auxdata
            self.auxdata_order.append(modifier_def['name'])
        return modifier

class hfpdf(object):
    def __init__(self, spec, **config_kwargs):
        self.config = modelconfig.from_spec(spec,**config_kwargs)
        self.spec = spec

    def _multiplicative_factors(self, channel, sample, pars):
        multiplicative_types = ['shapesys', 'normfactor', 'shapefactor','staterror']
        modifiers = [m['name'] for m in sample['modifiers'] if m['type'] in multiplicative_types]
        factors = []
        for m in modifiers:
            modifier, modpars = self.config.modifier(m), pars[self.config.par_slice(m)]
            mod_factor = modifier.apply(channel, sample, modpars)
            factors.append(mod_factor)
        return factors

    def _normsysfactor(self, channel, sample, pars):
        # normsysfactor(nom_sys_alphas)   = 1 + sum(interp(1, anchors[i][0],
        # anchors[i][0], val=alpha)  for i in range(nom_sys_alphas))
        modifiers = [m['name'] for m in sample['modifiers'] if m['type'] == 'normsys']
        factors = []
        for m in modifiers:
            modifier, modpars = self.config.modifier(m), pars[self.config.par_slice(m)]
            mod_factor = modifier.apply(channel, sample, modpars)
            factors.append(mod_factor)
        return tensorlib.product(factors)

    def _histosysdelta(self, channel, sample, pars):
        modifiers = [m['name'] for m in sample['modifiers'] if m['type'] == 'histosys']
        factors = []
        for m in modifiers:
            modifier, modpars = self.config.modifier(m), pars[self.config.par_slice(m)]
            mod_delta = modifier.apply(channel, sample, modpars)
            mod_factor = modifier.apply(channel, sample, modpars)
            factors.append(mod_factor)
        return tensorlib.sum(tensorlib.stack(factors), axis=0) if factors else None

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

        # print('sample factors {}'.format(factors))


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
