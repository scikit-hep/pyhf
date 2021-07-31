"""The pyhf command line interface."""
from pyhf.cli.cli import pyhf as cli
from pyhf.cli.rootio import cli as rootio
from pyhf.cli.spec import cli as spec
from pyhf.cli.infer import cli as infer
from pyhf.cli.complete import cli as complete
from pyhf.contrib import cli as contrib

__all__ = ['cli', 'rootio', 'spec', 'infer', 'complete', 'contrib']


def __dir__():
    return __all__
