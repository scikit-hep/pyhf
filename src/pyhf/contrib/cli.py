"""CLI for functionality that will get migrated out eventually."""
import logging
import click
from pathlib import Path

logging.basicConfig()
log = logging.getLogger(__name__)

__all__ = ["download"]


def __dir__():
    return __all__


@click.group(name="contrib")
def cli():
    """
    Contrib experimental operations.

    .. note::

        Requires installation of the ``contrib`` extra.

        .. code-block:: shell

            $ python -m pip install 'pyhf[contrib]'
    """
    from pyhf.contrib import utils  # Guard CLI from missing extra # noqa: F401


@cli.command()
@click.argument("archive-url")
@click.argument("output-directory")
@click.option("-v", "--verbose", is_flag=True, help="Enables verbose mode")
@click.option(
    "-f", "--force", is_flag=True, help="Force download from non-approved host"
)
@click.option(
    "-c",
    "--compress",
    is_flag=True,
    help="Keep the archive in a compressed tar.gz form",
)
def download(archive_url, output_directory, verbose, force, compress):
    """
    Download the patchset archive from the remote URL and extract it in a
    directory at the path given.

    Example:

    .. code-block:: shell

        $ pyhf contrib download --verbose https://doi.org/10.17182/hepdata.90607.v3/r3 1Lbb-likelihoods

        \b
        1Lbb-likelihoods/patchset.json
        1Lbb-likelihoods/README.md
        1Lbb-likelihoods/BkgOnly.json

    Raises:
        :class:`~pyhf.exceptions.InvalidArchiveHost`: if the provided archive host name is not known to be valid
    """
    try:
        from pyhf.contrib import utils

        utils.download(archive_url, output_directory, force, compress)

        if verbose:
            file_list = [str(file) for file in list(Path(output_directory).glob("*"))]
            print("\n".join(file_list))
    except AttributeError:
        log.error(
            "\nInstallation of the contrib extra is required to use the contrib CLI API"
            + "\nPlease install with: python -m pip install 'pyhf[contrib]'\n",
            exc_info=True,
        )
