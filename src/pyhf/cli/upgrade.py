"""The pyhf upgrade CLI subcommand."""
import logging

import click
import json

from pyhf.schema.upgrader import upgrade

log = logging.getLogger(__name__)


@click.group(name='upgrade')
def cli():
    """Operations for upgrading specifications."""


@cli.command()
@click.argument('workspace', default='-')
@click.option(
    '--version',
    help='The version to upgrade to',
    default=None,
)
@click.option(
    '--output-file',
    help='The location of the output json file. If not specified, prints to screen.',
    default=None,
)
def workspace(workspace, version, output_file):
    """
    Upgrade a HistFactory JSON workspace.
    """
    with click.open_file(workspace, 'r', encoding="utf-8") as specstream:
        spec = json.load(specstream)

    ws = upgrade(to_version=version).workspace(spec)

    if output_file is None:
        click.echo(json.dumps(ws, indent=4, sort_keys=True))
    else:
        with open(output_file, 'w+', encoding="utf-8") as out_file:
            json.dump(ws, out_file, indent=4, sort_keys=True)
        log.debug(f"Written to {output_file:s}")


@cli.command()
@click.argument('patchset', default='-')
@click.option(
    '--version',
    help='The version to upgrade to',
    default=None,
)
@click.option(
    '--output-file',
    help='The location of the output json file. If not specified, prints to screen.',
    default=None,
)
def patchset(patchset, version, output_file):
    """
    Upgrade a pyhf JSON PatchSet.
    """
    with click.open_file(patchset, 'r', encoding="utf-8") as specstream:
        spec = json.load(specstream)

    ps = upgrade(to_version=version).patchset(spec)

    if output_file is None:
        click.echo(json.dumps(ps, indent=4, sort_keys=True))
    else:
        with open(output_file, 'w+', encoding="utf-8") as out_file:
            json.dump(ps, out_file, indent=4, sort_keys=True)
        log.debug(f"Written to {output_file:s}")
