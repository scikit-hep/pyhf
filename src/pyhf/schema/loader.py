from pathlib import Path
import sys
import json
import pyhf.exceptions
from pyhf.schema.globals import SCHEMA_CACHE, SCHEMA_BASE, schemas

# importlib.resources.as_file wasn't added until Python 3.9?
if sys.version_info >= (3, 9):
    from importlib import resources
else:
    import importlib_resources as resources


def load_schema(schema_id):
    try:
        return SCHEMA_CACHE[f'{Path(SCHEMA_BASE).joinpath(schema_id)}']
    except KeyError:
        pass

    ref = schemas.joinpath(schema_id)
    with resources.as_file(ref) as path:
        if not path.exists():
            raise pyhf.exceptions.SchemaNotFound(
                f'The schema {schema_id} was not found. Do you have the right version or the right path? {path}'
            )
        with path.open() as json_schema:
            schema = json.load(json_schema)
            SCHEMA_CACHE[schema['$id']] = schema
        return SCHEMA_CACHE[schema['$id']]
