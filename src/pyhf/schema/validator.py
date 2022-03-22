import jsonschema
import pyhf.exceptions
from pyhf.schema.globals import SCHEMA_CACHE, SCHEMA_VERSION, schemas
from pyhf.schema.loader import load_schema


def validate(spec, schema_name, version=None):
    version = version or SCHEMA_VERSION

    schema = load_schema(f'{version}/{schema_name}')

    # note: trailing slash needed for RefResolver to resolve correctly
    resolver = jsonschema.RefResolver(
        base_uri=f"file://{schemas}/",
        referrer=f"{version}/{schema_name}",
        store=SCHEMA_CACHE,
    )
    validator = jsonschema.Draft6Validator(
        schema, resolver=resolver, format_checker=None
    )

    try:
        return validator.validate(spec)
    except jsonschema.ValidationError as err:
        raise pyhf.exceptions.InvalidSpecification(err, schema_name)
