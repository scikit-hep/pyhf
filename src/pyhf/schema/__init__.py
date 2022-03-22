import pathlib
import sys
from pyhf.schema.loader import load_schema
from pyhf.schema.validator import validate
from pyhf.schema import variables

__all__ = [
    "load_schema",
    "validate",
    "path",
    "version",
]


def __dir__():
    return __all__


class Schema(sys.modules[__name__].__class__):
    def __call__(self, new_path: pathlib.Path):
        variables.schemas = new_path

    @property
    def path(self):
        return variables.schemas

    @property
    def version(self):
        return variables.SCHEMA_VERSION


sys.modules[__name__].__class__ = Schema
