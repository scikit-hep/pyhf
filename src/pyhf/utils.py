import json
import jsonschema
import pkg_resources
from pathlib import Path
import yaml
import click
import hashlib

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
            base_uri=f"file://{pkg_resources.resource_filename(__name__, 'schemas/'):s}",
            referrer=schema_name,
            store=SCHEMA_CACHE,
        )
        validator = jsonschema.Draft6Validator(
            schema, resolver=resolver, format_checker=None
        )
        return validator.validate(spec)
    except jsonschema.ValidationError as err:
        raise InvalidSpecification(err, schema_name)


def options_from_eqdelimstring(opts):
    document = '\n'.join(
        f"{opt.split('=', 1)[0]}: {opt.split('=', 1)[1]}" for opt in opts
    )
    return yaml.full_load(document)


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


def remove_prefix(text, prefix):
    """
    Remove a prefix from the beginning of the provided text.

    Example:

        >>> import pyhf
        >>> pyhf.utils.remove_prefix("alpha_syst1", "alpha_")
        'syst1'

    Args:
        text (:obj:`str`): A provided input to manipulate.
        prefix (:obj:`str`): A prefix to remove from provided input, if it exists.

    Returns:
        stripped_text (:obj:`str`): Text with the prefix removed.
    """
    # NB: python3.9 can be `return text.removeprefix(prefix)`
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def citation(oneline=False):
    """
    Get the bibtex citation for pyhf

    Example:

        >>> import pyhf
        >>> pyhf.utils.citation(True)
        '@software{pyhf,  author = {Lukas Heinrich and Matthew Feickert and Giordon Stark},  title = "{pyhf: v0.6.1}",  version = {0.6.1},  doi = {10.5281/zenodo.1169739},  url = {https://github.com/scikit-hep/pyhf},}@article{pyhf_joss,  doi = {10.21105/joss.02823},  url = {https://doi.org/10.21105/joss.02823},  year = {2021},  publisher = {The Open Journal},  volume = {6},  number = {58},  pages = {2823},  author = {Lukas Heinrich and Matthew Feickert and Giordon Stark and Kyle Cranmer},  title = {pyhf: pure-Python implementation of HistFactory statistical models},  journal = {Journal of Open Source Software}}'

    Keyword Args:
        oneline (:obj:`bool`): Whether to provide citation with new lines (default) or as a one-liner.

    Returns:
        citation (:obj:`str`): The citation for this software
    """
    path = Path(
        pkg_resources.resource_filename(
            __name__, str(Path('data').joinpath('citation.bib'))
        )
    )
    with path.open() as fp:
        # remove end-of-file newline if there is one
        data = fp.read().strip()

    if oneline:
        data = ''.join(data.splitlines())
    return data
