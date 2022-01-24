import json
import jsonschema
from pathlib import Path
import yaml
import click
import hashlib
import sys

if sys.version_info >= (3, 9):
    from importlib import resources
else:
    import importlib_resources as resources

from pyhf.exceptions import InvalidSpecification

SCHEMA_CACHE = {}
SCHEMA_BASE = "https://scikit-hep.org/pyhf/schemas/"
SCHEMA_VERSION = '1.0.0'

__all__ = [
    "EqDelimStringParamType",
    "citation",
    "digest",
    "load_schema",
    "options_from_eqdelimstring",
    "validate",
]


def __dir__():
    return __all__


def load_schema(schema_id):
    global SCHEMA_CACHE
    try:
        return SCHEMA_CACHE[f'{Path(SCHEMA_BASE).joinpath(schema_id)}']
    except KeyError:
        pass

    ref = resources.files('pyhf') / 'schemas' / schema_id
    with resources.as_file(ref) as path:
        with path.open() as json_schema:
            schema = json.load(json_schema)
            SCHEMA_CACHE[schema['$id']] = schema
        return SCHEMA_CACHE[schema['$id']]


# load the defs.json as it is included by $ref
# load_schema('defs.json')


def validate(spec, schema_name, version=None):
    version = version or SCHEMA_VERSION
    schema = load_schema(f'{version}/{schema_name}')

    # note: trailing slash needed for RefResolver to resolve correctly
    resolver = jsonschema.RefResolver(
        base_uri=f"file://{resources.files('pyhf') / 'schemas' / version / schema_name}",
        referrer=schema_name,
        store=SCHEMA_CACHE,
    )
    validator = jsonschema.Draft6Validator(
        schema, resolver=resolver, format_checker=None
    )

    try:
        return validator.validate(spec)
    except jsonschema.ValidationError as err:
        raise InvalidSpecification(err, schema_name)


def options_from_eqdelimstring(opts):
    document = '\n'.join(
        f"{opt.split('=', 1)[0]}: {opt.split('=', 1)[1]}" for opt in opts
    )
    return yaml.safe_load(document)


class EqDelimStringParamType(click.ParamType):
    name = 'equal-delimited option'

    def convert(self, value, param, ctx):
        try:
            return options_from_eqdelimstring([value])
        except IndexError:
            self.fail(f'{value:s} is not a valid equal-delimited string', param, ctx)


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
        obj (:obj:`jsonable`): A JSON-serializable object to compute the digest of. Usually a :class:`~pyhf.workspace.Workspace` object.
        algorithm (:obj:`str`): The hashing algorithm to use.

    Returns:
        digest (:obj:`str`): The digest for the JSON-serialized object provided and hash algorithm specified.
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


def citation(oneline=False):
    """
    Get the bibtex citation for pyhf

    Example:

        >>> import pyhf
        >>> pyhf.utils.citation(oneline=True)
        '@software{pyhf,  author = {Lukas Heinrich and Matthew Feickert and Giordon Stark},  title = "{pyhf: v0.6.3}",  version = {0.6.3},  doi = {10.5281/zenodo.1169739},  url = {https://doi.org/10.5281/zenodo.1169739},  note = {https://github.com/scikit-hep/pyhf/releases/tag/v0.6.3}}@article{pyhf_joss,  doi = {10.21105/joss.02823},  url = {https://doi.org/10.21105/joss.02823},  year = {2021},  publisher = {The Open Journal},  volume = {6},  number = {58},  pages = {2823},  author = {Lukas Heinrich and Matthew Feickert and Giordon Stark and Kyle Cranmer},  title = {pyhf: pure-Python implementation of HistFactory statistical models},  journal = {Journal of Open Source Software}}'

    Keyword Args:
        oneline (:obj:`bool`): Whether to provide citation with new lines (default) or as a one-liner.

    Returns:
        citation (:obj:`str`): The citation for this software
    """
    ref = resources.files('pyhf') / 'data' / 'citation.bib'
    with resources.as_file(ref) as path:
        data = path.read_text().strip()

    if oneline:
        data = ''.join(data.splitlines())
    return data
