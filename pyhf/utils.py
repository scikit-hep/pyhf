import json, jsonschema
import pkg_resources

from .exceptions import InvalidSpecification

DEFAULT_SCHEMA = pkg_resources.resource_filename(__name__,'data/spec.json')
def get_default_schema():
  global DEFAULT_SCHEMA
  return DEFAULT_SCHEMA

SCHEMA_CACHE = {}
def load_schema(schema):
    global SCHEMA_CACHE
    try:
        return SCHEMA_CACHE[schema]
    except KeyError:
        pass

    SCHEMA_CACHE[schema] = json.load(open(schema))
    return SCHEMA_CACHE[schema]

def validate(spec, schema):
    schema = load_schema(schema)
    try:
        return jsonschema.validate(spec, schema)
    except jsonschema.ValidationError as err:
        raise InvalidSpecification(err)
