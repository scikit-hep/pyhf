"""Helper utilities for common tasks."""

from urllib.parse import urlparse
import tarfile
from io import BytesIO
import logging
from .. import exceptions

logging.basicConfig()
log = logging.getLogger(__name__)

try:
    import requests

    def download(archive_url, output_directory, force=False, compress=False):
        """
        Download the patchset archive from the remote URL and extract it in a
        directory at the path given.

        Example:

            >>> from pyhf.contrib.utils import download
            >>> download("https://doi.org/10.17182/hepdata.90607.v3/r3", "1Lbb-likelihoods")
            >>> import os
            >>> sorted(os.listdir("1Lbb-likelihoods"))
            ['BkgOnly.json', 'README.md', 'patchset.json']
            >>> download("https://doi.org/10.17182/hepdata.90607.v3/r3", "1Lbb-likelihoods.tar.gz", compress=True)
            >>> import glob
            >>> glob.glob("1Lbb-likelihoods.tar.gz")
            ['1Lbb-likelihoods.tar.gz']

        Args:
            archive_url (:obj:`str`): The URL of the :class:`~pyhf.patchset.PatchSet` archive to download.
            output_directory (:obj:`str`): Name of the directory to unpack the archive into.
            force (:obj:`bool`): Force download from non-approved host. Default is ``False``.
            compress (:obj:`bool`): Keep the archive in a compressed ``tar.gz`` form. Default is ``False``.

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
            if compress:
                with open(output_directory, "wb") as archive:
                    archive.write(response.content)
            else:
                with tarfile.open(
                    mode="r|gz", fileobj=BytesIO(response.content)
                ) as archive:
                    archive.extractall(output_directory)


except ModuleNotFoundError as excep:
    log.error(
        str(excep)
        + "\nInstallation of the contrib extra is required to use pyhf.contrib.utils.download"
        + "\nPlease install with: python -m pip install pyhf[contrib]\n"
    )
