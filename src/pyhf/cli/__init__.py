"""The pyhf command line interface."""
from .cli import pyhf as cli
from .rootio import cli as rootio
from .spec import cli as spec
from .infer import cli as infer

__all__ = ['cli', 'rootio', 'spec', 'infer']
