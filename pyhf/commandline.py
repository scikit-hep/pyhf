import logging
logging.basicConfig()

import click
import json
from . import readxml

@click.command()
@click.option('--entrypoint-xml', required=True, prompt='Top-level XML', help='The top-level XML file for the workspace definition.')
@click.option('--workspace', required=True, prompt='Workspace directory', help='The location of workspace.')
def xml2json(entrypoint_xml, workspace):
    spec = readxml(entrypoint_xml, workspace)
    import pdb; pdb.set_trace()
