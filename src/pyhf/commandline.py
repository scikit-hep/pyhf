import logging

import click

from .version import __version__
from . import cli

logging.basicConfig()
log = logging.getLogger(__name__)


@click.group(context_settings=dict(help_option_names=['-h', '--help']))
@click.version_option(version=__version__)
def pyhf():
    pass


pyhf.add_command(cli.stats)
pyhf.add_command(cli.rootio)
pyhf.add_command(cli.spec)
