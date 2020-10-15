"""The pyhf command line interface."""
from .cli import pyhf as cli
from .rootio import cli as rootio
from .spec import cli as spec
from .infer import cli as infer
from .complete import cli as complete
from ..contrib import cli as contrib

__all__ = ['cli', 'rootio', 'spec', 'infer', 'complete', 'contrib']
