import sys
from pyhf.schema.loader import load_schema
from pyhf.schema.validator import validate
from pyhf.schema.globals import schemas as path, SCHEMA_VERSION as version

__all__ = [
    "load_schema",
    "validate",
    "path",
    "version",
]


def __dir__():
    return __all__


class Schema(sys.modules[__name__].__class__):
    def __call__(self):
        return 42


sys.modules[__name__].__class__ = Schema
