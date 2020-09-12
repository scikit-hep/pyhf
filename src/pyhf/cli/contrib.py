"""CLI for functionality that will get migrated out eventually."""
import logging

import click
from urllib.parse import urlparse
import requests
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

        $ pyhf contrib download --verbose https://www.hepdata.net/record/resource/1408476?view=true 3L-likelihoods

        \b
        3L-likelihoods/patchset.json
        3L-likelihoods/README.md
        3L-likelihoods/BkgOnly.json

    Raises:
        :class:`~pyhf.exceptions.InvalidArchiveHost`: if the provided archive host name is not known to be valid
    """
    if not force:
        valid_hosts = ["www.hepdata.net", "doi.org"]
        netloc = urlparse(archive_url).netloc
        if netloc not in valid_hosts:
            raise exceptions.InvalidArchiveHost(
                f"{netloc} is not an approved archive host: {', '.join(str(host) for host in valid_hosts)}\n"
                + "To download an archive from this host use the --force option."
            )

    with requests.get(archive_url) as response:
        with tarfile.open(mode="r|gz", fileobj=BytesIO(response.content)) as archive:
            archive.extractall(output_directory)

    if verbose:
        file_list = [str(file) for file in list(Path(output_directory).glob("*"))]
        print("\n".join(file_list))
