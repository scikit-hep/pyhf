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
        >>> import pyhf
        >>> pyhf.get_backend()
        (<pyhf.tensor.numpy_backend.numpy_backend object at 0x...>, <pyhf.optimize.opt_scipy.scipy_optimizer object at 0x...>)

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
        >>> import pyhf
        >>> import tensorflow as tf
        >>> pyhf.set_backend(pyhf.tensor.tensorflow_backend(session=tf.Session()))
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

    def expected_sample(self, channel, sample, pars):
        """
        The idea is that we compute all bin-values at once.. each bin is a product of various factors, but sum are per-channel the other per-channel

            b1 = shapesys_1   |      shapef_1   |
            b2 = shapesys_2   |      shapef_2   |
            ...             normfac1    ..     normfac2
                            (broad)            (broad)
            bn = shapesys_n   |      shapef_1   |

        this can be achieved by `numpy`'s `broadcast_arrays` and `np.product`. The broadcast expands the scalars or one-length arrays to an array which we can then uniformly multiply

            >>> import numpy as np
            >>> np.broadcast_arrays([2],[3,4,5],[6],[7,8,9])
            [array([2, 2, 2]), array([3, 4, 5]), array([6, 6, 6]), array([7, 8, 9])]
            >>> ## also
            >>> np.broadcast_arrays(2,[3,4,5],6,[7,8,9])
            [array([2, 2, 2]), array([3, 4, 5]), array([6, 6, 6]), array([7, 8, 9])]
            >>> ## also
            >>> factors = [2,[3,4,5],6,[7,8,9]]
            >>> np.broadcast_arrays(*factors)
            [array([2, 2, 2]), array([3, 4, 5]), array([6, 6, 6]), array([7, 8, 9])]

        So that something like

            >>> import numpy as np
            >>> np.product(np.broadcast_arrays([2],[3,4,5],[6],[7,8,9]),axis=0)
            array([252, 384, 540])

        which is just `[ 2*3*6*7, 2*4*6*8, 2*5*6*9]`.

        Notice how some factors (for fixed channel c and sample s) depend on
        bin b and some don't (eq 6 CERN-OPEN-2012-016). The broadcasting lets
        you scale all bins the same way, such as when you have a ttbar
        normalization factor that scales all bins.

        Shape === affects each bin separately
        Non-shape === affects all bins the same way (just changes normalization, keeps shape the same)
        """
        # for each sample the expected ocunts are
        # counts = (multiplicative factors) * (normsys multiplier) * (histsys delta + nominal hist)
        #        = f1*f2*f3*f4* nomsysfactor(nom_sys_alphas) * hist(hist_addition(histosys_alphas) + nomdata)
        # nomsysfactor(nom_sys_alphas)   = 1 + sum(interp(1, anchors[i][0], anchors[i][0], val=alpha)  for i in range(nom_sys_alphas))
        # hist_addition(histosys_alphas) = sum(interp(nombin, anchors[i][0],
        # anchors[i][0], val=alpha) for i in range(histosys_alphas))
        #
        # Formula:
        #     \nu_{cb} (\phi_p, \alpha_p, \gamma_p) = \lambda_{cs} \gamma_{cb} \phi_{cs}(\alpha) \eta_{cs}(\alpha) \sigma_{csb}(\alpha)
        # \gamma == statsys, shapefactor
        # \phi == normfactor, overallsys
        # \sigma == histosysdelta + nominal

        # first, collect the factors from all modifiers
        results = {'shapesys': [],
                   'normfactor': [],
                   'shapefactor': [],
                   'staterror': [],
                   'histosys': [],
                   'normsys': []}
        for m in sample['modifiers']:
            modifier, modpars = self.config.modifier(m['name']), pars[self.config.par_slice(m['name'])]
            results[m['type']].append(modifier.apply(channel, sample, modpars))


        # start building the entire set of factors
        factors = []
        # scalars that get broadcasted to shape of vectors
        factors += results['normfactor']
        factors += results['normsys']
        # vectors
        factors += results['shapesys']
        factors += results['shapefactor']
        factors += results['staterror']

        nominal = tensorlib.astensor(sample['data'])
        factors += [tensorlib.sum(tensorlib.stack([
          nominal,
          tensorlib.sum(tensorlib.stack(results['histosys']), axis=0)
        ]), axis=0) if len(results['histosys']) > 0 else nominal]

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
    r"""
    The test statistic, :math:`q_{\mu}`, for establishing an upper
    limit on the strength parameter, :math:`\mu`, as defiend in
    Equation (14) in `arXiv:1007.1727`_ .

    .. _`arXiv:1007.1727`: https://arxiv.org/abs/1007.1727

    .. math::
       :nowrap:

       \begin{equation}
          q_{\mu} = \left\{\begin{array}{ll}
          -2\ln\lambda\left(\mu\right), &\hat{\mu} < \mu,\\
          0, & \hat{\mu} > \mu
          \end{array}\right.
        \end{equation}


    Args:
        mu (Number or Tensor): The signal strength parameter
        data (Tensor): The data to be considered
        pdf (Tensor): The model used in the likelihood ratio calculation
        init_pars (Tensor): The initial parameters
        par_bounds(Tensor): The bounds on the paramter values

    Returns:
        Float: The calculated test statistic, :math:`q_{\mu}`
    """
    mubhathat = optimizer.constrained_bestfit(
        loglambdav, mu, data, pdf, init_pars, par_bounds)
    muhatbhat = optimizer.unconstrained_bestfit(
        loglambdav, data, pdf, init_pars, par_bounds)
    qmu = loglambdav(mubhathat, data, pdf) - loglambdav(muhatbhat, data, pdf)
    qmu = tensorlib.where(muhatbhat[pdf.config.poi_index] > mu, [0], qmu)
    return qmu


def pvals_from_teststat(sqrtqmu_v, sqrtqmuA_v):
    r"""
    The :math:`p`-values for signal strength :math:`\mu` and Asimov strength :math:`\mu'`
    as defined in Equations (59) and (57) of `arXiv:1007.1727`_

    .. _`arXiv:1007.1727`: https://arxiv.org/abs/1007.1727

    .. math::

        p_{\mu} = 1-F\left(q_{\mu}\middle|\mu'\right) = 1- \Phi\left(q_{\mu} - \frac{\left(\mu-\mu'\right)}{\sigma}\right)

    with Equation (29)

    .. math::

        \frac{(\mu-\mu')}{\sigma} = \sqrt{\Lambda}= \sqrt{q_{\mu,A}}

    given the observed test statistics :math:`q_{\mu}` and :math:`q_{\mu,A}`.

    Args:
        sqrtqmu_v (Number or Tensor): The root of the calculated test statistic, :math:`\sqrt{q_{\mu}}`
        sqrtqmuA_v (Number or Tensor): The root of the calculated test statistic given the Asimov data, :math:`\sqrt{q_{\mu,A}}`

    Returns:
        Tuple of Floats: The :math:`p`-values for the signal + background, background only, and signal only hypotheses respectivley
    """
    CLsb = 1 - tensorlib.normal_cdf(sqrtqmu_v)
    CLb = 1 - tensorlib.normal_cdf(sqrtqmu_v - sqrtqmuA_v)
    CLs = CLsb / CLb
    return CLsb, CLb, CLs


def runOnePoint(muTest, data, pdf, init_pars = None, par_bounds = None):
    r"""
    Computes test statistics (and expected statistics) for a single value
    of the parameter of interest

    Args:
        muTest (Number or Tensor): The value of the parameter of interest (POI)
        data (Number or Tensor): The root of the calculated test statistic given the Asimov data, :math:`\sqrt{q_{\mu,A}}`
        init_pars (Array or Tensor): the initial parameter values to be used for minimization
        par_bounds (Array or Tensor): the parameter value bounds to be used for minimization

    Returns:
        Tuple of Floats: a tuple containing (qmu, qmu_A, CLsb, CLb, CLs, CLs_exp)
                         where qmu and qmu_A are the test statistics for the
                         observed and Asimov datasets respectively.
                         CLsb, CLb are the signal + background and background-only p-values
                         CLs is the modified p-value
                         CLs_exp is a 5-tuple of expected CLs values at percentiles
                         of the background-only test-statistics corresponding to
                         percentiles of the normal distribution for
                         (-2,-1,0,1,2) :math:`\sigma`
    """

    init_pars = init_pars or pdf.config.suggested_init()
    par_bounds = par_bounds or pdf.config.suggested_bounds()


    asimov_mu = 0.
    asimov_data = generate_asimov_data(
        asimov_mu, data, pdf, init_pars, par_bounds)

    qmu_v = tensorlib.clip(
        qmu(muTest, data, pdf, init_pars, par_bounds), 0, max=None)
    sqrtqmu_v = tensorlib.sqrt(qmu_v)

    qmuA_v = tensorlib.clip(
        qmu(muTest, asimov_data, pdf, init_pars, par_bounds), 0, max=None)
    sqrtqmuA_v = tensorlib.sqrt(qmuA_v)

    CLsb, CLb, CLs = pvals_from_teststat(sqrtqmu_v, sqrtqmuA_v)

    CLs_exp = []
    for nsigma in [-2, -1, 0, 1, 2]:
        sqrtqmu_v_sigma = sqrtqmuA_v - nsigma
        CLs_exp.append(
            pvals_from_teststat(sqrtqmu_v_sigma, sqrtqmuA_v)[-1])
    CLs_exp = tensorlib.astensor(CLs_exp)
    return qmu_v, qmuA_v, CLsb, CLb, CLs, CLs_exp
