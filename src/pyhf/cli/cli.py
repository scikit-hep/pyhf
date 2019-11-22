import logging

import click

from ..version import __version__
from . import rootio, spec, stats

logging.basicConfig()
log = logging.getLogger(__name__)


@click.group(context_settings=dict(help_option_names=['-h', '--help']))
@click.version_option(version=__version__)
def pyhf():
    pass


pyhf.add_command(rootio.cli)
pyhf.add_command(spec.cli)
pyhf.add_command(stats.cli)
