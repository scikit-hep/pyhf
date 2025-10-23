"""The pyhf command line interface."""

from pyhf.cli.cli import pyhf as cli
from pyhf.cli.complete import cli as complete
from pyhf.cli.infer import cli as infer
from pyhf.cli.rootio import cli as rootio
from pyhf.cli.spec import cli as spec
from pyhf.contrib import cli as contrib

__all__ = ['cli', 'complete', 'contrib', 'infer', 'rootio', 'spec']


def __dir__():
    return __all__
