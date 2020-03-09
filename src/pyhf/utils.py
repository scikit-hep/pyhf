import json
import jsonschema
import pkg_resources
from pathlib import Path
import yaml
import click

from .exceptions import InvalidSpecification

SCHEMA_CACHE = {}
SCHEMA_BASE = "https://scikit-hep.org/pyhf/schemas/"
SCHEMA_VERSION = '1.0.0'


def load_schema(schema_id, version=None):
    global SCHEMA_CACHE
    if not version:
        version = SCHEMA_VERSION
    try:
        return SCHEMA_CACHE[f'{SCHEMA_BASE}{Path().joinpath(version, schema_id)}']
    except KeyError:
        pass

    path = pkg_resources.resource_filename(
        __name__, Path().joinpath('schemas', version, schema_id).as_posix()
    )
    with open(path) as json_schema:
        schema = json.load(json_schema)
        SCHEMA_CACHE[schema['$id']] = schema
    return SCHEMA_CACHE[schema['$id']]


# load the defs.json as it is included by $ref
load_schema('defs.json')


def validate(spec, schema_name, version=None):
    schema = load_schema(schema_name, version=version)
    try:
        resolver = jsonschema.RefResolver(
            base_uri='file://{0:s}'.format(
                pkg_resources.resource_filename(__name__, 'schemas/')
            ),
            referrer=schema_name,
            store=SCHEMA_CACHE,
        )
        validator = jsonschema.Draft6Validator(
            schema, resolver=resolver, format_checker=None
        )
        return validator.validate(spec)
    except jsonschema.ValidationError as err:
        raise InvalidSpecification(err)


def options_from_eqdelimstring(opts):
    document = '\n'.join('{0}: {1}'.format(*opt.split('=', 1)) for opt in opts)
    return yaml.full_load(document)


class EqDelimStringParamType(click.ParamType):
    name = 'equal-delimited option'

    def convert(self, value, param, ctx):
        try:
            return options_from_eqdelimstring([value])
        except IndexError:
            self.fail(
                '{0:s} is not a valid equal-delimited string'.format(value), param, ctx
            )
