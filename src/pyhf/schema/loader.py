from pathlib import Path
import sys
import json
import pyhf.exceptions
from pyhf.schema import variables

# importlib.resources.as_file wasn't added until Python 3.9
# c.f. https://docs.python.org/3.9/library/importlib.html#importlib.resources.as_file
if sys.version_info >= (3, 9):
    from importlib import resources
else:
    import importlib_resources as resources


def load_schema(schema_id: str):
    """
    Get a schema by relative path from cache, or load it into the cache and return.

    Args:
        schema_id (str): Relative path to schema from :attr:`pyhf.schema.path`

    Returns:
        schema (dict): The loaded schema.
    """
    try:
        return variables.SCHEMA_CACHE[
            f'{Path(variables.SCHEMA_BASE).joinpath(schema_id)}'
        ]
    except KeyError:
        pass

    ref = variables.schemas.joinpath(schema_id)
    with resources.as_file(ref) as path:
        if not path.exists():
            raise pyhf.exceptions.SchemaNotFound(
                f'The schema {schema_id} was not found. Do you have the right version or the right path? {path}'
            )
        with path.open() as json_schema:
            schema = json.load(json_schema)
            variables.SCHEMA_CACHE[schema['$id']] = schema
        return variables.SCHEMA_CACHE[schema['$id']]


def load_schema(schema_id, version=None):
    """
    Load a version of a schema, referenced by its identifier.

    Args:
        schema_id (:obj:`string`): The name of a schema to validate against.
        version (:obj:`string`): The version of the schema to use. If not set, the default will be the latest and greatest schema supported by this library. Default: ``None``.

    Raises:
        FileNotFoundError: if the provided ``schema_id`` cannot be found.

    Returns:
        :obj:`dict`: The loaded schema.

    Example:
        >>> import pyhf
        >>> schema = pyhf.utils.load_schema('defs.json')
        >>> type(schema)
        <class 'dict'>
        >>> schema.keys()
        dict_keys(['$schema', '$id', 'definitions'])
        >>> pyhf.utils.load_schema('defs.json', version='0.0.0')
        Traceback (most recent call last):
            ...
        FileNotFoundError: ...
    """
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



