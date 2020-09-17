"""CLI for functionality that will get migrated out eventually."""
import logging

import click
from urllib.parse import urlparse
import tarfile
from io import BytesIO
from pathlib import Path

from .. import exceptions

logging.basicConfig()
log = logging.getLogger(__name__)


@click.group(name="contrib")
def cli():
    """
    Contrib experimental operations.

    .. note::

        Requires installation of the ``contrib`` extra.

        .. code-block:: shell

            $ python -m pip install pyhf[contrib]
    """


try:
    import requests

    @cli.command()
    @click.argument("archive-url", default="-")
    @click.argument("output-directory", default="-")
    @click.option("-v", "--verbose", is_flag=True, help="Enables verbose mode")
    @click.option(
        "-f", "--force", is_flag=True, help="Force download from non-approved host"
    )
    def download(archive_url, output_directory, verbose, force):
        """
        Download the patchset archive from the remote URL and extract it in a
        directory at the path given.

        Example:

        .. code-block:: shell

            $ pyhf contrib download --verbose https://www.hepdata.net/record/resource/1408476?view=true 1Lbb-likelihoods

            \b
            1Lbb-likelihoods/patchset.json
            1Lbb-likelihoods/README.md
            1Lbb-likelihoods/BkgOnly.json

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
            with tarfile.open(
                mode="r|gz", fileobj=BytesIO(response.content)
            ) as archive:
                archive.extractall(output_directory)

        if verbose:
            file_list = [str(file) for file in list(Path(output_directory).glob("*"))]
            print("\n".join(file_list))


except ModuleNotFoundError as excep:
    exception_info = (
        str(excep)
        + "\nInstallation of the contrib extra is required to use the contrib CLI API"
        + "\nPlease install with: python -m pip install pyhf[contrib]\n"
    )
    print(exception_info)
