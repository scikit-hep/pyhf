from pathlib import Path
import sys
import json
import pyhf.exceptions
from pyhf.schema import variables

# importlib.resources.as_file wasn't added until Python 3.9?
if sys.version_info >= (3, 9):
    from importlib import resources
else:
    import importlib_resources as resources


def load_schema(schema_id):
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
