import json, jsonschema
import pkg_resources

from .exceptions import InvalidSpecification

def get_default_schema():
    r"""
    Returns the absolute filepath default schema for pyhf. This usually points
    to pyhf/data/spec.json.

    Returns:
        Schema File Path: a string containing the absolute path to the default
                          schema file.
    """
    return pkg_resources.resource_filename(__name__,'data/spec.json')


SCHEMA_CACHE = {}
def load_schema(schema):
    global SCHEMA_CACHE
    try:
        return SCHEMA_CACHE[schema]
    except KeyError:
        pass

    SCHEMA_CACHE[schema] = json.load(open(schema))
    return SCHEMA_CACHE[schema]


def validate(spec, schema):
    schema = load_schema(schema)
    try:
        return jsonschema.validate(spec, schema)
    except jsonschema.ValidationError as err:
        raise InvalidSpecification(err)


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
