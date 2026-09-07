import json
from importlib import resources

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
    # Keyed by the relative path rather than the schema $id, so that custom
    # schemas under pyhf.schema.path with relative $ids also hit the cache.
    try:
        return variables.SCHEMA_CACHE[schema_id]
    except KeyError:
        pass

    ref = variables.schemas.joinpath(schema_id)
    with resources.as_file(ref) as path:
        if not path.exists():
            msg = f"The schema {schema_id} was not found. Do you have the right version or the right path? {path}"
            raise pyhf.exceptions.SchemaNotFound(msg)
        with path.open(encoding="utf-8") as json_schema:
            schema = json.load(json_schema)
    variables.SCHEMA_CACHE[schema_id] = schema
    return schema


# pre-populate the cache to avoid network access
# on first validation in standard usage
# (not in pyhf.schema.variables to avoid circular imports)
load_schema(f"{variables.SCHEMA_VERSION}/defs.json")
