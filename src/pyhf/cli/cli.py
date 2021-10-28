"""The pyhf Command Line Interface."""
import logging
from typing import Optional

import click
import typer

from pyhf import __version__
from pyhf.cli import rootio, spec, infer, patchset, complete
from pyhf.contrib import cli as contrib
from pyhf import utils

logging.basicConfig()
log = logging.getLogger(__name__)

app = typer.Typer()


@app.command()
def top():
    """
    Top level command, form Typer
    """
    typer.echo("The Typer app is at the top level")


def _version_callback(value: bool):
    if value:
        typer.echo(f"pyhf, v{__version__}")
        raise typer.Exit()


def _print_citation(ctx, value):
    if not value or ctx.resilient_parsing:
        return
    typer.echo(utils.citation())
    raise typer.Exit()


@app.callback()
def callback(
    version: Optional[bool] = typer.Option(
        None, "--version", callback=_version_callback, is_eager=True
    ),
    citation: Optional[bool] = typer.Option(
        None,
        "--cite",
        "--citation",
        help="Print the BibTeX citation for this software.",
        callback=_print_citation,
        is_eager=True,
        is_flag=True,  # Needed?
        expose_value=False,  # Needed?
    ),
):
    """
    Typer app, including Click subapp

    Top-level CLI entrypoint.
    """


@click.group(context_settings=dict(help_option_names=['-h', '--help']))
def pyhf():
    """Top-level CLI entrypoint."""


typer_click_object = typer.main.get_command(app)

typer_click_object.add_command(pyhf)

# typer_click_object.add_command(rootio.cli)
typer_click_object.add_command(rootio.json2xml)
typer_click_object.add_command(rootio.xml2json)

# typer_click_object.add_command(spec.cli)
typer_click_object.add_command(spec.inspect)
typer_click_object.add_command(spec.prune)
typer_click_object.add_command(spec.rename)
typer_click_object.add_command(spec.combine)
typer_click_object.add_command(spec.digest)
typer_click_object.add_command(spec.sort)

# typer_click_object.add_command(infer.cli)
typer_click_object.add_command(infer.fit)
typer_click_object.add_command(infer.cls)

typer_click_object.add_command(patchset.cli)

typer_click_object.add_command(complete.cli)

typer_click_object.add_command(contrib.cli)
