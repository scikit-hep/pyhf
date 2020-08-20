"""The pyhf Command Line Interface."""
import logging

import click

from ..version import __version__
from . import rootio, spec, infer, patchset, complete

logging.basicConfig()
log = logging.getLogger(__name__)


@click.group(context_settings=dict(help_option_names=['-h', '--help']))
@click.version_option(version=__version__)
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
pyhf.add_command(infer.cls)

pyhf.add_command(patchset.cli)

pyhf.add_command(complete.cli)
