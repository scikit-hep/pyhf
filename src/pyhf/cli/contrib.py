"""CLI for functionality that will get migrated out eventually."""
import logging

import click
import urllib.request
import tarfile
from io import BytesIO
from pathlib import Path

from .. import exceptions

logging.basicConfig()
log = logging.getLogger(__name__)


@click.group(name="contrib")
def cli():
    """Contrib experimental operations"""


@cli.command()
@click.argument("archive-url", default="-")
@click.argument("output-directory", default="-")
@click.option("-v", "--verbose", is_flag=True, help="Enables verbose mode")
@click.option(
    "-f", "--force", is_flag=True, help="Force download from non-approved host"
)
def download(archive_url, output_directory, verbose, force):
    """
    Download the patchset archive from the remote URL and extract it in a directory at the path given.

    Example:

    .. code-block:: shell

        $ pyhf contrib download --verbose https://www.hepdata.net/record/resource/1408476?view=true 3L-likelihood

        \b
        3L-likelihoods/patchset.json
        3L-likelihoods/README.md
        3L-likelihoods/BkgOnly.json

    Raises:
        :class:`~pyhf.exceptions.InvalidArchiveHost`: if the provided archive host name is not known to be valid
    """
    if not force:
        hostname = archive_url.split("://")[1].split("/")[0].strip("www.")
        valid_hosts = ["hepdata.net", "doi.org"]
        if hostname not in valid_hosts:
            raise exceptions.InvalidArchiveHost(
                f"{hostname} is not an approved archive host: {', '.join(str(host) for host in valid_hosts)}\n"
                + "To download an archive from this host use the --force option."
            )

    req = urllib.request.Request(archive_url)
    with urllib.request.urlopen(req) as response:
        with tarfile.open(mode="r|gz", fileobj=BytesIO(response.read())) as archive:
            archive.extractall(output_directory)

    if verbose:
        file_list = [str(file) for file in list(Path(output_directory).glob("*"))]
        print("\n".join(file_list))
