"""The pyhf utils CLI subcommand."""

import logging
from pathlib import Path

import click

from pyhf import utils

log = logging.getLogger(__name__)


@click.group(name="utils")
def cli():
    """Utils CLI group."""


@cli.command()
@click.option(
    "-o",
    "--output-file",
    help="The location of the output file. If not specified, prints to screen.",
    default=None,
)
def environment(output_file):
    """Print information about the pyhf environment useful for bug reports."""
    info = utils.environment_info()

    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with output_file.open("w+", encoding="utf-8") as out_file:
            out_file.write(info)
        log.debug("Written to %s", output_file)
    else:
        click.echo(info)
