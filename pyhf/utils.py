import json
import jsonschema
import pkg_resources

from .exceptions import InvalidSpecification
from . import get_backend


def get_default_schema():
    r"""
    Returns the absolute filepath default schema for pyhf. This usually points
    to pyhf/data/spec.json.

    Returns:
        Schema File Path: a string containing the absolute path to the default
                          schema file.
    """
    return pkg_resources.resource_filename(__name__, 'data/spec.json')


SCHEMA_CACHE = {}


def load_schema(schema):
    global SCHEMA_CACHE
    try:
        return SCHEMA_CACHE[schema]
    except KeyError:
        pass

    with open(schema) as json_schema:
        SCHEMA_CACHE[schema] = json.load(json_schema)
    return SCHEMA_CACHE[schema]


def validate(spec, schema):
    schema = load_schema(schema)
    try:
        return jsonschema.validate(spec, schema)
    except jsonschema.ValidationError as err:
        raise InvalidSpecification(err)


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
    tensorlib, optimizer = get_backend()
    mubhathat = optimizer.constrained_bestfit(
        loglambdav, mu, data, pdf, init_pars, par_bounds
    )
    muhatbhat = optimizer.unconstrained_bestfit(
        loglambdav, data, pdf, init_pars, par_bounds
    )
    qmu = loglambdav(mubhathat, data, pdf) - loglambdav(muhatbhat, data, pdf)
    qmu = tensorlib.where(muhatbhat[pdf.config.poi_index] > mu, [0], qmu)
    return qmu


def generate_asimov_data(asimov_mu, data, pdf, init_pars, par_bounds):
    _, optimizer = get_backend()
    bestfit_nuisance_asimov = optimizer.constrained_bestfit(
        loglambdav, asimov_mu, data, pdf, init_pars, par_bounds
    )
    return pdf.expected_data(bestfit_nuisance_asimov)


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
    tensorlib, _ = get_backend()
    CLsb = 1 - tensorlib.normal_cdf(sqrtqmu_v)
    CLb = 1 - tensorlib.normal_cdf(sqrtqmu_v - sqrtqmuA_v)
    CLs = CLsb / CLb
    return CLsb, CLb, CLs


def hypotest(poi_test, data, pdf, init_pars=None, par_bounds=None, **kwargs):
    r"""
    Computes test statistics (and expected statistics) for a single value of the parameter of interest

    Args:
        poi_test (Number or Tensor): The value of the parameter of interest (POI)
        data (Number or Tensor): The root of the calculated test statistic given the Asimov data, :math:`\sqrt{q_{\mu,A}}`
        init_pars (Array or Tensor): The initial parameter values to be used for minimization
        par_bounds (Array or Tensor): The parameter value bounds to be used for minimization

    Keyword Args:
        return_observed_set (bool): Bool for returning :math:`CL_{s+b}` and :math:`CL_{b}`
        return_expected (bool): Bool for returning :math:`CL_{\textrm{exp}}`
        return_expected_set (bool): Bool for returning the :math:`(-2,-1,0,1,2)\sigma` :math:`CL_{\textrm{exp}}` --- the "Brazil band"
        return_test_statistics (bool): Bool for returning :math:`q_{\mu}` and :math:`q_{\mu,A}`

    Returns:
        Tuple of Floats and lists of Floats:

            :math:`CL_{s}`: The modified :math:`p`-value

            :math:`\left[CL_{s+b}, CL_{b}\right]`: The signal + background and background-only :math:`p`-values. Only returned when :code:`return_observed_set` is :code:`True`.

            :math:`CL_{\textrm{exp}}`: The expected :math:`CL_{s}` value corresponding to the background-only. Only returned when :code:`return_expected` is :code:`True`.

            :math:`\left(CL_{\textrm{exp}}\right)`: 5-tuple of expected :math:`CL_{s}` values at percentiles of the background-only test-statistics corresponding to percentiles of the normal distribution for :math:`(-2,-1,0,1,2)\sigma`. Also known as the "Brazil band". Only returned when :code:`return_expected_set` is :code:`True`.

            :math:`\left[q_{\mu}, q_{\mu,A}\right]`: The test statistics for the observed and Asimov datasets respectively. Only returned when :code:`return_test_statistics` is :code:`True`.
    """

    init_pars = init_pars or pdf.config.suggested_init()
    par_bounds = par_bounds or pdf.config.suggested_bounds()
    tensorlib, _ = get_backend()

    asimov_mu = 0.0
    asimov_data = generate_asimov_data(asimov_mu, data, pdf, init_pars, par_bounds)

    qmu_v = tensorlib.clip(qmu(poi_test, data, pdf, init_pars, par_bounds), 0, max=None)
    sqrtqmu_v = tensorlib.sqrt(qmu_v)

    qmuA_v = tensorlib.clip(
        qmu(poi_test, asimov_data, pdf, init_pars, par_bounds), 0, max=None
    )
    sqrtqmuA_v = tensorlib.sqrt(qmuA_v)

    CLsb, CLb, CLs = pvals_from_teststat(sqrtqmu_v, sqrtqmuA_v)

    _returns = [CLs]
    if kwargs.get('return_observed_set'):
        _returns.append([CLsb, CLb])
    if kwargs.get('return_expected_set'):
        CLs_exp = []
        for n_sigma in [-2, -1, 0, 1, 2]:
            sqrtqmu_v_sigma = sqrtqmuA_v - n_sigma
            CLs_exp.append(pvals_from_teststat(sqrtqmu_v_sigma, sqrtqmuA_v)[-1])
        CLs_exp = tensorlib.astensor(CLs_exp)
        if kwargs.get('return_expected'):
            _returns.append(CLs_exp[2])
        _returns.append(CLs_exp)
    elif kwargs.get('return_expected'):
        _returns.append(pvals_from_teststat(sqrtqmuA_v, sqrtqmuA_v)[-1])
    if kwargs.get('return_test_statistics'):
        _returns.append([qmu_v, qmuA_v])

    # Enforce a consistent return type of the observed CLs
    if len(_returns) > 1:
        _returns = tuple(_returns)
    else:
        _returns = _returns[0]

    return _returns
