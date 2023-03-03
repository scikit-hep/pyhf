"""The pyhf utils CLI subcommand."""
import logging

import click
from pyhf import utils

log = logging.getLogger(__name__)


@click.group(name="utils")
def cli():
    """Utils CLI group."""


@cli.command()
def debug():
    click.echo(utils.debug_info())
