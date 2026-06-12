import json
from importlib import resources
from pathlib import Path

import pyhf.exceptions
from pyhf.schema import variables


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
            f"{Path(variables.SCHEMA_BASE).joinpath(schema_id)}"
        ]
    except KeyError:
        pass

    schema = read_schema(schema_id)
    variables.SCHEMA_CACHE[schema["$id"]] = schema
    return variables.SCHEMA_CACHE[schema["$id"]]


def read_schema(schema_id: str):
    """
    Read a schema by relative path directly from disk, bypassing the cache.

    Unlike :func:`load_schema`, this does not consult or mutate
    :data:`pyhf.schema.variables.SCHEMA_CACHE`. It is used to resolve
    cross-schema ``$ref``\\ s during validation without polluting the cache.

    Args:
        schema_id (str): Relative path to schema from :attr:`pyhf.schema.path`

    Returns:
        schema (dict): The loaded schema.

    Raises:
        ~pyhf.exceptions.SchemaNotFound: if the provided ``schema_id`` cannot be found.
    """
    ref = variables.schemas.joinpath(schema_id)
    with resources.as_file(ref) as path:
        if not path.exists():
            msg = f"The schema {schema_id} was not found. Do you have the right version or the right path? {path}"
            raise pyhf.exceptions.SchemaNotFound(msg)
        with path.open(encoding="utf-8") as json_schema:
            return json.load(json_schema)


# pre-populate the cache to avoid network access
# on first validation in standard usage
# (not in pyhf.schema.variables to avoid circular imports)
load_schema(f"{variables.SCHEMA_VERSION}/defs.json")
