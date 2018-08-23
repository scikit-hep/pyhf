import logging
logging.basicConfig()
log = logging.getLogger(__name__)

import click
import json
from . import readxml

@click.command()
@click.option('--entrypoint-xml', required=True, prompt='Top-level XML', help='The top-level XML file for the workspace definition.', type=click.Path(exists=True))
@click.option('--workspace', required=True, prompt='Workspace directory', help='The location of workspace.', type=click.Path(exists=True))
@click.option('--output-file', required=True, prompt='Output file', help='The location of the output json file. If not specified, prints to screen.', type=click.Path(exists=False))
def xml2json(entrypoint_xml, workspace, output_file=None):
    spec = readxml.parse(entrypoint_xml, workspace, enable_tqdm=True)
    json.dump(spec, open(output_file, 'w+'), indent=4, sort_keys=True)
    log.info("Written to {0:s}".format(output_file))
