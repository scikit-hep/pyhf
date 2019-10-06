import json
import jsonschema
import pkg_resources
import os
import yaml
import click

from .exceptions import InvalidSpecification
from . import get_backend

SCHEMA_CACHE = {}
SCHEMA_BASE = "https://diana-hep.org/pyhf/schemas/"
SCHEMA_VERSION = '1.0.0'


def load_schema(schema_id, version=None):
    global SCHEMA_CACHE
    if not version:
        version = SCHEMA_VERSION
    try:
        return SCHEMA_CACHE[
            "{0:s}{1:s}".format(SCHEMA_BASE, os.path.join(version, schema_id))
        ]
    except KeyError:
        pass

    path = pkg_resources.resource_filename(
        __name__, os.path.join('schemas', version, schema_id)
    )
    with open(path) as json_schema:
        schema = json.load(json_schema)
        SCHEMA_CACHE[schema['$id']] = schema
    return SCHEMA_CACHE[schema['$id']]


# load the defs.json as it is included by $ref
load_schema('defs.json')


def validate(spec, schema_name, version=None):
    schema = load_schema(schema_name, version=version)
    try:
        resolver = jsonschema.RefResolver(
            base_uri='file://{0:s}'.format(
                pkg_resources.resource_filename(__name__, 'schemas/')
            ),
            referrer=schema_name,
            store=SCHEMA_CACHE,
        )
        validator = jsonschema.Draft6Validator(
            schema, resolver=resolver, format_checker=None
        )
        return validator.validate(spec)
    except jsonschema.ValidationError as err:
        raise InvalidSpecification(err)


def options_from_eqdelimstring(opts):
    document = '\n'.join('{0}: {1}'.format(*opt.split('=', 1)) for opt in opts)
    return yaml.full_load(document)


class EqDelimStringParamType(click.ParamType):
    name = 'equal-delimited option'

    def convert(self, value, param, ctx):
        try:
            return options_from_eqdelimstring([value])
        except IndexError:
            self.fail(
                '{0:s} is not a valid equal-delimited string'.format(value), param, ctx
            )


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
        pdf (|pyhf.pdf.Model|_): The HistFactory statistical model used in the likelihood ratio calculation
        init_pars (Tensor): The initial parameters
        par_bounds(Tensor): The bounds on the paramter values

    .. |pyhf.pdf.Model| replace:: ``pyhf.pdf.Model``
    .. _pyhf.pdf.Model: https://diana-hep.org/pyhf/_generated/pyhf.pdf.Model.html

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
    qmu = tensorlib.where(
        muhatbhat[pdf.config.poi_index] > mu, tensorlib.astensor([0]), qmu
    )
    return qmu


def generate_asimov_data(asimov_mu, data, pdf, init_pars, par_bounds):
    _, optimizer = get_backend()
    bestfit_nuisance_asimov = optimizer.constrained_bestfit(
        loglambdav, asimov_mu, data, pdf, init_pars, par_bounds
    )
    return pdf.expected_data(bestfit_nuisance_asimov)


def pvals_from_teststat(sqrtqmu_v, sqrtqmuA_v, qtilde=False):
    r"""
    The :math:`p`-values for signal strength :math:`\mu` and Asimov strength :math:`\mu'` as defined in Equations (59) and (57) of `arXiv:1007.1727`_

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
        qtilde (Bool): When ``True`` perform the calculation using the alternative test statistic, :math:`\tilde{q}`, as defined in Equation (62) of `arXiv:1007.1727`_

    Returns:
        Tuple of Floats: The :math:`p`-values for the signal + background, background only, and signal only hypotheses respectivley
    """
    tensorlib, _ = get_backend()
    if not qtilde:  # qmu
        nullval = sqrtqmu_v
        altval = -(sqrtqmuA_v - sqrtqmu_v)
    else:  # qtilde
        if sqrtqmu_v < sqrtqmuA_v:
            nullval = sqrtqmu_v
            altval = -(sqrtqmuA_v - sqrtqmu_v)
        else:
            qmu = tensorlib.power(sqrtqmu_v, 2)
            qmu_A = tensorlib.power(sqrtqmuA_v, 2)
            nullval = (qmu + qmu_A) / (2 * sqrtqmuA_v)
            altval = (qmu - qmu_A) / (2 * sqrtqmuA_v)
    CLsb = 1 - tensorlib.normal_cdf(nullval)
    CLb = 1 - tensorlib.normal_cdf(altval)
    CLs = CLsb / CLb
    return CLsb, CLb, CLs


def pvals_from_teststat_expected(sqrtqmuA_v, nsigma=0):
    r"""
    Computes the expected :math:`p`-values CLsb, CLb and CLs for data corresponding to a given percentile of the alternate hypothesis.

    Args:
        sqrtqmuA_v (Number or Tensor): The root of the calculated test statistic given the Asimov data, :math:`\sqrt{q_{\mu,A}}`
        nsigma (Number or Tensor): The number of standard deviations of variations of the signal strength from the background only hypothesis :math:`\left(\mu=0\right)`

    Returns:
        Tuple of Floats: The :math:`p`-values for the signal + background, background only, and signal only hypotheses respectivley
    """

    # NOTE:
    # To compute the expected p-value, one would need to first compute a hypothetical
    # observed test-statistic for a dataset whose best-fit value is mu^ = mu'-n*sigma:
    # $q_n$, and the proceed with the normal p-value computation for whatever test-statistic
    # was used. However, we can make a shortcut by just computing the p-values in mu^/sigma
    # space, where the p-values are Clsb = cdf(x-sqrt(lambda)) and CLb=cdf(x)

    tensorlib, _ = get_backend()
    CLsb = tensorlib.normal_cdf(nsigma - sqrtqmuA_v)
    CLb = tensorlib.normal_cdf(nsigma)
    CLs = CLsb / CLb
    return CLsb, CLb, CLs


def hypotest(
    poi_test, data, pdf, init_pars=None, par_bounds=None, qtilde=False, **kwargs
):
    r"""
    Computes :math:`p`-values and test statistics for a single value of the parameter of interest

    Args:
        poi_test (Number or Tensor): The value of the parameter of interest (POI)
        data (Number or Tensor): The root of the calculated test statistic given the Asimov data, :math:`\sqrt{q_{\mu,A}}`
        pdf (|pyhf.pdf.Model|_): The HistFactory statistical model
        init_pars (Array or Tensor): The initial parameter values to be used for minimization
        par_bounds (Array or Tensor): The parameter value bounds to be used for minimization
        qtilde (Bool): When ``True`` perform the calculation using the alternative test statistic, :math:`\tilde{q}`, as defined in Equation (62) of `arXiv:1007.1727`_

    .. |pyhf.pdf.Model| replace:: ``pyhf.pdf.Model``
    .. _pyhf.pdf.Model: https://diana-hep.org/pyhf/_generated/pyhf.pdf.Model.html

    Keyword Args:
        return_tail_probs (bool): Bool for returning :math:`\textrm{CL}_{s+b}` and :math:`\textrm{CL}_{b}`
        return_expected (bool): Bool for returning :math:`\textrm{CL}_{\textrm{exp}}`
        return_expected_set (bool): Bool for returning the :math:`(-2,-1,0,1,2)\sigma` :math:`\textrm{CL}_{\textrm{exp}}` --- the "Brazil band"
        return_test_statistics (bool): Bool for returning :math:`q_{\mu}` and :math:`q_{\mu,A}`

    Returns:
        Tuple of Floats and lists of Floats:

            - :math:`\textrm{CL}_{s}`: The :math:`p`-value compared to the given threshold :math:`\alpha`, typically taken to be :math:`0.05`, defined in `arXiv:1007.1727`_ as

            .. _`arXiv:1007.1727`: https://arxiv.org/abs/1007.1727

            .. math::

                \textrm{CL}_{s} = \frac{\textrm{CL}_{s+b}}{\textrm{CL}_{b}} = \frac{p_{s+b}}{1-p_{b}}

            to protect against excluding signal models in which there is little sensitivity. In the case that :math:`\textrm{CL}_{s} \leq \alpha` the given signal model is excluded.

            - :math:`\left[\textrm{CL}_{s+b}, \textrm{CL}_{b}\right]`: The signal + background :math:`p`-value and 1 minus the background only :math:`p`-value as defined in Equations (75) and (76) of `arXiv:1007.1727`_

            .. math::

                \textrm{CL}_{s+b} = p_{s+b} = \int\limits_{q_{\textrm{obs}}}^{\infty} f\left(q\,\middle|s+b\right)\,dq = 1 - \Phi\left(\frac{q_{\textrm{obs}} + 1/\sigma_{s+b}^{2}}{2/\sigma_{s+b}}\right)

            .. math::

                \textrm{CL}_{b} = 1- p_{b} = 1 - \int\limits_{-\infty}^{q_{\textrm{obs}}} f\left(q\,\middle|b\right)\,dq = 1 - \Phi\left(\frac{q_{\textrm{obs}} - 1/\sigma_{b}^{2}}{2/\sigma_{b}}\right)

            with Equations (73) and (74) for the mean

            .. math::

                E\left[q\right] = \frac{1 - 2\mu}{\sigma^{2}}

            and variance

            .. math::

                V\left[q\right] = \frac{4}{\sigma^{2}}

            of the test statistic :math:`q` under the background only and and signal + background hypotheses. Only returned when ``return_tail_probs`` is ``True``.

            - :math:`\textrm{CL}_{s,\textrm{exp}}`: The expected :math:`\textrm{CL}_{s}` value corresponding to the test statistic under the background only hypothesis :math:`\left(\mu=0\right)`. Only returned when ``return_expected`` is ``True``.

            - :math:`\textrm{CL}_{s,\textrm{exp}}` band: The set of expected :math:`\textrm{CL}_{s}` values corresponding to the median significance of variations of the signal strength from the background only hypothesis :math:`\left(\mu=0\right)` at :math:`(-2,-1,0,1,2)\sigma`. That is, the :math:`p`-values that satisfy Equation (89) of `arXiv:1007.1727`_

            .. math::

                \textrm{band}_{N\sigma} = \mu' + \sigma\,\Phi^{-1}\left(1-\alpha\right) \pm N\sigma

            for :math:`\mu'=0` and :math:`N \in \left\{-2, -1, 0, 1, 2\right\}`. These values define the boundaries of an uncertainty band sometimes referred to as the "Brazil band". Only returned when ``return_expected_set`` is ``True``.

            - :math:`\left[q_{\mu}, q_{\mu,A}\right]`: The test statistics for the observed and Asimov datasets respectively. Only returned when ``return_test_statistics`` is ``True``.
    """

    init_pars = init_pars or pdf.config.suggested_init()
    par_bounds = par_bounds or pdf.config.suggested_bounds()
    tensorlib, _ = get_backend()

    asimov_mu = 0.0
    asimov_data = generate_asimov_data(asimov_mu, data, pdf, init_pars, par_bounds)

    qmu_v = tensorlib.clip(
        qmu(poi_test, data, pdf, init_pars, par_bounds), 0, max_value=None
    )
    sqrtqmu_v = tensorlib.sqrt(qmu_v)

    qmuA_v = tensorlib.clip(
        qmu(poi_test, asimov_data, pdf, init_pars, par_bounds), 0, max_value=None
    )
    sqrtqmuA_v = tensorlib.sqrt(qmuA_v)

    CLsb, CLb, CLs = pvals_from_teststat(sqrtqmu_v, sqrtqmuA_v, qtilde=qtilde)

    _returns = [CLs]
    if kwargs.get('return_tail_probs'):
        _returns.append([CLsb, CLb])
    if kwargs.get('return_expected_set'):
        CLs_exp = []
        for n_sigma in [-2, -1, 0, 1, 2]:
            CLs_exp.append(pvals_from_teststat_expected(sqrtqmuA_v, nsigma=n_sigma)[-1])
        CLs_exp = tensorlib.astensor(CLs_exp)
        if kwargs.get('return_expected'):
            _returns.append(CLs_exp[2])
        _returns.append(CLs_exp)
    elif kwargs.get('return_expected'):
        _returns.append(pvals_from_teststat_expected(sqrtqmuA_v)[-1])
    if kwargs.get('return_test_statistics'):
        _returns.append([qmu_v, qmuA_v])
    # Enforce a consistent return type of the observed CLs
    return tuple(_returns) if len(_returns) > 1 else _returns[0]
