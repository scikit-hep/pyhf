"""The pyhf spec CLI subcommand."""
import logging

import click
import json

from pyhf.schema.upgrader import upgrade_workspace, upgrade_patchset

log = logging.getLogger(__name__)


@click.group(name='upgrade')
def cli():
    """Operations for upgrading specifications."""


@cli.command()
@click.argument('workspace', default='-')
@click.option(
    '--output-file',
    help='The location of the output json file. If not specified, prints to screen.',
    default=None,
)
def workspace(workspace, output_file):
    """
    Upgrade a HistFactory JSON workspace.
    """
    with click.open_file(workspace, 'r', encoding="utf-8") as specstream:
        spec = json.load(specstream)

    ws = upgrade_workspace(spec)

    if output_file is None:
        click.echo(json.dumps(ws, indent=4, sort_keys=True))
    else:
        with open(output_file, 'w+', encoding="utf-8") as out_file:
            json.dump(ws, out_file, indent=4, sort_keys=True)
        log.debug(f"Written to {output_file:s}")


@cli.command()
@click.argument('patchset', default='-')
@click.option(
    '--output-file',
    help='The location of the output json file. If not specified, prints to screen.',
    default=None,
)
def patchset(patchset, output_file):
    """
    Upgrade a pyhf JSON PatchSet.
    """
    with click.open_file(patchset, 'r', encoding="utf-8") as specstream:
        spec = json.load(specstream)

    ps = upgrade_patchset(spec)

    if output_file is None:
        click.echo(json.dumps(ps, indent=4, sort_keys=True))
    else:
        with open(output_file, 'w+', encoding="utf-8") as out_file:
            json.dump(ps, out_file, indent=4, sort_keys=True)
        log.debug(f"Written to {output_file:s}")
