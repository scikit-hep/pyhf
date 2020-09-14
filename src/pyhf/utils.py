import json
import jsonschema
import pkg_resources
from pathlib import Path
import yaml
import click
import hashlib
import ast

from .exceptions import InvalidSpecification

SCHEMA_CACHE = {}
SCHEMA_BASE = "https://scikit-hep.org/pyhf/schemas/"
SCHEMA_VERSION = '1.0.0'


def load_schema(schema_id, version=None):
    global SCHEMA_CACHE
    if not version:
        version = SCHEMA_VERSION
    try:
        return SCHEMA_CACHE[f'{SCHEMA_BASE}{Path(version).joinpath(schema_id)}']
    except KeyError:
        pass

    path = pkg_resources.resource_filename(
        __name__, str(Path('schemas').joinpath(version, schema_id))
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


def digest(obj, algorithm='sha256'):
    """
    Get the digest for the provided object. Note: object must be JSON-serializable.

    The hashing algorithms supported are in :mod:`hashlib`, part of Python's Standard Libraries.

    Example:

        >>> import pyhf
        >>> obj = {'a': 2.0, 'b': 3.0, 'c': 1.0}
        >>> pyhf.utils.digest(obj)
        'a38f6093800189b79bc22ef677baf90c75705af2cfc7ff594159eca54eaa7928'
        >>> pyhf.utils.digest(obj, algorithm='md5')
        '2c0633f242928eb55c3672fed5ba8612'
        >>> pyhf.utils.digest(obj, algorithm='sha1')
        '49a27f499e763766c9545b294880df277be6f545'

    Raises:
        ValueError: If the object is not JSON-serializable or if the algorithm is not supported.

    Args:
        obj (`obj`): A JSON-serializable object to compute the digest of. Usually a :class:`~pyhf.workspace.Workspace` object.
        algorithm (`str`): The hashing algorithm to use.

    Returns:
        digest (`str`): The digest for the JSON-serialized object provided and hash algorithm specified.
    """

    try:
        stringified = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode('utf8')
    except TypeError:
        raise ValueError(
            "The supplied object is not JSON-serializable for calculating a hash."
        )
    try:
        hash_alg = getattr(hashlib, algorithm)
    except AttributeError:
        raise ValueError(
            f"{algorithm} is not an algorithm provided by Python's hashlib library."
        )
    return hash_alg(stringified).hexdigest()


def parse_parameter_name(expression):
    """
    Convert a parameter name expression to the corresponding parameter index.

    Returns:
        par_name (`str`): parameter name
        slice (`slice`): slice/indexing to use
    """
    _error = lambda reason: ValueError(
        f"{expression} is not a valid parameter name expression. {reason}"
    )

    tree = ast.parse(expression)
    expr = tree.body[0].value
    if len(tree.body) > 1 or not isinstance(expr, (ast.Name, ast.Subscript)):
        raise _error("Expression is too complex.")

    # provided par_name="{name:s}"
    if isinstance(expr, ast.Name):
        par_name = expr.id
        return par_name, slice(None)

    if not isinstance(expr.value, ast.Name):
        raise _error("Additional subscripts are not supported.")

    par_name = expr.value.id
    # provided par_name="{name:s}[index]"
    if isinstance(expr.slice, ast.Index):
        if not isinstance(expr.slice.value, ast.Num):
            raise _error("Index must be a number.")

        # provided par_name="{name:s}[{index:num}]"
        return par_name, expr.slice.value.n

    if isinstance(expr.slice, ast.Slice):
        # provided par_name="{name:s}[slice]"
        if not (
            (expr.slice.lower is None or isinstance(expr.slice.lower, ast.Num))
            and (expr.slice.upper is None or isinstance(expr.slice.upper, ast.Num))
            and (expr.slice.step is None or isinstance(expr.slice.step, ast.Num))
        ):
            raise _error("Indices must be non-negative numbers.")

        # provided par_name="{name:s}[{index:num}:{index:num}:{index:num}]"
        lower = getattr(expr.slice.lower, 'n', None)
        upper = getattr(expr.slice.upper, 'n', None)
        step = getattr(expr.slice.step, 'n', None)
        return par_name, slice(lower, upper, step)
