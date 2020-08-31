"""The pyhf spec CLI subcommand."""
import logging

import click
import json

from ..patchset import PatchSet
from ..workspace import Workspace

logging.basicConfig()
log = logging.getLogger(__name__)


@click.group(name='patchset')
def cli():
    """Operations involving patchsets."""


@cli.command()
@click.argument('patchset', default='-')
@click.option('--name', help='The name of the patch to extract.', default=None)
@click.option(
    '--output-file',
    help='The location of the output json file. If not specified, prints to screen.',
    default=None,
)
@click.option(
    '--with-metadata/--without-metadata',
    default=False,
    help="Include patchset metadata in output.",
)
def extract(patchset, name, output_file, with_metadata):
    """
    Extract a patch from a patchset.

    Raises:
        :class:`~pyhf.exceptions.InvalidPatchLookup`: if the provided patch name is not in the patchset

    Returns:
        jsonpatch (:obj:`list`): A list of jsonpatch operations to apply to a workspace.
    """
    with click.open_file(patchset, 'r') as fstream:
        patchset_spec = json.load(fstream)

    patchset = PatchSet(patchset_spec)
    patch = patchset[name]

    if with_metadata:
        result = {'metadata': patch.metadata, 'patch': patch.patch}
        result['metadata'].update(patchset.metadata)
    else:
        result = patch.patch

    if output_file:
        with open(output_file, 'w+') as out_file:
            json.dump(result, out_file, indent=4, sort_keys=True)
        log.debug("Written to {0:s}".format(output_file))
    else:
        click.echo(json.dumps(result, indent=4, sort_keys=True))


@cli.command()
@click.argument('background-only', default='-')
@click.argument('patchset', default='-')
@click.option('--name', help='The name of the patch to extract.', default=None)
@click.option(
    '--output-file',
    help='The location of the output json file. If not specified, prints to screen.',
    default=None,
)
def apply(background_only, patchset, name, output_file):
    """
    Apply a patch from patchset to the background-only workspace specification.

    Raises:
        :class:`~pyhf.exceptions.InvalidPatchLookup`: if the provided patch name is not in the patchset
        :class:`~pyhf.exceptions.PatchSetVerificationError`: if the patchset cannot be verified against the workspace specification

    Returns:
        workspace (:class:`~pyhf.workspace.Workspace`): The patched background-only workspace.
    """
    with click.open_file(background_only, 'r') as specstream:
        spec = json.load(specstream)

    ws = Workspace(spec)

    with click.open_file(patchset, 'r') as fstream:
        patchset_spec = json.load(fstream)

    patchset = PatchSet(patchset_spec)
    patched_ws = patchset.apply(ws, name)

    if output_file:
        with open(output_file, 'w+') as out_file:
            json.dump(patched_ws, out_file, indent=4, sort_keys=True)
        log.debug("Written to {0:s}".format(output_file))
    else:
        click.echo(json.dumps(patched_ws, indent=4, sort_keys=True))


@cli.command()
@click.argument('background-only', default='-')
@click.argument('patchset', default='-')
def verify(background_only, patchset):
    """
    Verify the patchset digests against a background-only workspace specification. Verified if no exception was raised.

    Raises:
        :class:`~pyhf.exceptions.PatchSetVerificationError`: if the patchset cannot be verified against the workspace specification

    Returns:
        None
    """
    with click.open_file(background_only, 'r') as specstream:
        spec = json.load(specstream)

    ws = Workspace(spec)

    with click.open_file(patchset, 'r') as fstream:
        patchset_spec = json.load(fstream)

    patchset = PatchSet(patchset_spec)
    patchset.verify(ws)

    click.echo("All good.")
