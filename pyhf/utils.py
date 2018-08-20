import json, jsonschema

from .exceptions import InvalidSpecification

SCHEMA_CACHE = {}
def load_schema(schema):
  global SCHEMA_CACHE
  try:
    return SCHEMA_CACHE[schema]
  except KeyError:
    pass

  try:
    SCHEMA_CACHE[schema] = json.load(open(schema))
    return SCHEMA_CACHE[schema]
  except:
    raise

def validate(spec, schema):
  schema = load_schema(schema)
  try:
    return jsonschema.validate(spec, schema)
  except jsonschema.ValidationError as err:
    raise InvalidSpecification(err)
