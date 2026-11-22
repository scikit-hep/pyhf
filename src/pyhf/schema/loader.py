import json
from importlib import resources

import pyhf.exceptions
from pyhf.schema import variables


def _schema_cache_key(schema_id: str) -> str:
    """Return the :data:`~pyhf.schema.variables.SCHEMA_CACHE` key for ``schema_id``.

    Keys are paths relative to :attr:`pyhf.schema.path`. A URI under
    :data:`~pyhf.schema.variables.SCHEMA_BASE` (typically a schema ``$id``) is
    reduced to that relative path so both forms share one cache entry.

    Do not build the key with :class:`pathlib.Path`:
    ``Path(SCHEMA_BASE).joinpath(schema_id)`` collapses ``https://`` to
    ``https:/`` and never matches the stored ``$id``
    (https://github.com/scikit-hep/pyhf/issues/2749).
    """
    if schema_id.startswith(variables.SCHEMA_BASE):
        return schema_id.removeprefix(variables.SCHEMA_BASE)
    return schema_id


def load_schema(schema_id: str):
    """
    Get a schema by relative path from cache, or load it into the cache and return.

    Args:
        schema_id (str): Relative path to schema from :attr:`pyhf.schema.path`.
         A URI under :data:`~pyhf.schema.variables.SCHEMA_BASE` is also accepted
         and looks up the same cached object.

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
    # Keyed by the relative path rather than pathlib(SCHEMA_BASE) or $id, so
    # that bundled schemas and custom schemas with relative $ids both hit.
    cache_key = _schema_cache_key(schema_id)
    try:
        return variables.SCHEMA_CACHE[cache_key]
    except KeyError:
        pass

    ref = variables.schemas.joinpath(cache_key)
    with resources.as_file(ref) as path:
        if not path.exists():
            msg = f"The schema {schema_id} was not found. Do you have the right version or the right path? {path}"
            raise pyhf.exceptions.SchemaNotFound(msg)
        with path.open(encoding="utf-8") as json_schema:
            schema = json.load(json_schema)
    variables.SCHEMA_CACHE[cache_key] = schema
    return schema


# pre-populate the cache to avoid network access
# on first validation in standard usage
# (not in pyhf.schema.variables to avoid circular imports)
load_schema(f"{variables.SCHEMA_VERSION}/defs.json")
