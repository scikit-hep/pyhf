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

    Example:
        >>> import pyhf
        >>> schema = pyhf.schema.load_schema("1.0.0/defs.json")
        >>> type(schema)
        <class 'dict'>
        >>> schema.keys()
        dict_keys(['$schema', '$id', 'definitions'])
        >>> pyhf.schema.load_schema("0.0.0/defs.json")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        pyhf.exceptions.SchemaNotFound: ...

    Returns:
        schema (dict): The loaded schema.

    Raises:
        ~pyhf.exceptions.SchemaNotFound: if the provided ``schema_id`` cannot be found.
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
        with path.open(encoding="utf-8") as json_schema:
            schema = json.load(json_schema)
            variables.SCHEMA_CACHE[schema['$id']] = schema
        return variables.SCHEMA_CACHE[schema['$id']]


# pre-populate the cache to avoid network access
# on first validation in standard usage
# (not in pyhf.schema.variables to avoid circular imports)
load_schema(f'{variables.SCHEMA_VERSION}/defs.json')
