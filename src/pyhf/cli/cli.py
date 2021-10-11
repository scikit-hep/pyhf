"""The pyhf Command Line Interface."""
import logging

import click

from pyhf import __version__
from pyhf.cli import rootio, spec, infer, patchset, complete
from pyhf.contrib import cli as contrib
from pyhf import utils

logging.basicConfig()
log = logging.getLogger(__name__)


def _print_citation(ctx, param, value):
    if not value or ctx.resilient_parsing:
        return
    click.echo(utils.citation())
    ctx.exit()


@click.group(context_settings=dict(help_option_names=['-h', '--help']))
@click.version_option(version=__version__)
@click.option(
    "--cite",
    "--citation",
    help="Print the bibtex citation for this software",
    default=False,
    is_flag=True,
    callback=_print_citation,
    expose_value=False,
    is_eager=True,
)
def pyhf():
    """Top-level CLI entrypoint."""


# pyhf.add_command(rootio.cli)
pyhf.add_command(rootio.json2xml)
pyhf.add_command(rootio.xml2json)

# pyhf.add_command(spec.cli)
pyhf.add_command(spec.inspect)
pyhf.add_command(spec.prune)
pyhf.add_command(spec.rename)
pyhf.add_command(spec.combine)
pyhf.add_command(spec.digest)
pyhf.add_command(spec.sort)

# pyhf.add_command(infer.cli)
pyhf.add_command(infer.fit)
pyhf.add_command(infer.cls)

pyhf.add_command(patchset.cli)

pyhf.add_command(complete.cli)

pyhf.add_command(contrib.cli)
