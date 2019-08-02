import json
import jsonschema
import pkg_resources
import os
import yaml
import click

from .exceptions import InvalidSpecification
from . import get_backend

SCHEMA_CACHE = {}
SCHEMA_BASE = "https://scikit-hep.org/pyhf/schemas/"
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

        def _true_case():
            nullval = sqrtqmu_v
            altval = -(sqrtqmuA_v - sqrtqmu_v)
            return nullval, altval

        def _false_case():
            qmu = tensorlib.power(sqrtqmu_v, 2)
            qmu_A = tensorlib.power(sqrtqmuA_v, 2)
            nullval = (qmu + qmu_A) / (2 * sqrtqmuA_v)
            altval = (qmu - qmu_A) / (2 * sqrtqmuA_v)
            return nullval, altval

        nullval, altval = tensorlib.conditional(
            (sqrtqmu_v < sqrtqmuA_v)[0], _true_case, _false_case
        )
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
