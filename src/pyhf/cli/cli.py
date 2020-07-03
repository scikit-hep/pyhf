"""The pyhf Command Line Interface."""
import logging

import click

try:
    import click_completion
    click_completion.init()
except ImportError:
    pass


from ..version import __version__
from . import rootio, spec, infer, patchset

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

# pyhf.add_command(infer.cli)
pyhf.add_command(infer.cls)

pyhf.add_command(patchset.cli)

try:
    @pyhf.command(help = 'generate shell completion code')
    @click.argument('shell', required=False, type=click_completion.DocumentedChoice(click_completion.core.shells))
    def shell_completion(shell):
        click.echo(click_completion.core.get_code(shell))
except NameError:
    pass