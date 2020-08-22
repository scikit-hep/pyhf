"""The pyhf spec CLI subcommand."""
import logging

import click
import json
import subprocess

from .. import exceptions
from ..patchset import PatchSet
from ..workspace import Workspace

logging.basicConfig()
log = logging.getLogger(__name__)


@click.group(name='patchset')
def cli():
    """Operations involving patchsets."""


@cli.command()
@click.argument("archive-url", default="-")
@click.argument("output-directory", default="-")
@click.option("-v", "--verbose", is_flag=True, help="Enables verbose mode")
@click.option(
    "-f", "--force", is_flag=True, help="Force download from non-approved repository"
)
def download(archive_url, output_directory, verbose, force):
    """
    Download the patchset archive from the remote URL and extract it in a directory at the path given.

    Returns:
        None
    """
    if not force:
        hostname = archive_url.split("://")[1].split("/")[0].strip("www.")
        valid_hosts = ["hepdata.net", "doi.org"]
        if hostname not in valid_hosts:
            raise exceptions.InvalidArchiveHost(
                f"{hostname} is not an approved archive host: {', '.join(str(host) for host in valid_hosts)}\n"
                + "To download an archive from this host use the --force option."
            )

    curl_cmd = ["curl", "-sL", archive_url]
    tar_options = "xzv" if verbose else "xz"
    tar_cmd = ["tar", f"-{tar_options}", f"--one-top-level={output_directory}"]
    ps = subprocess.Popen(curl_cmd, stdout=subprocess.PIPE)
    output = subprocess.run(tar_cmd, stdin=ps.stdout, capture_output=True)
    # output = subprocess.run(tar_cmd, stdin=ps.stdout, stdout=subprocess.STDOUT)
    ps.wait()


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
