import logging
logging.basicConfig()
log = logging.getLogger(__name__)

import click
import json
from . import readxml

@click.group(context_settings=dict(help_option_names=['-h', '--help']))
def pyhf():
    pass

@pyhf.command()
@click.option('--entrypoint-xml', required=True, prompt='Top-level XML', help='The top-level XML file for the PDF definition.', type=click.Path(exists=True))
@click.option('--basedir', required=True, prompt='Base directory', help='The base directory for the XML files to point relative to.', type=click.Path(exists=True))
@click.option('--output-file', required=True, prompt='Output file', help='The location of the output json file. If not specified, prints to screen.')
@click.option('--tqdm/--no-tqdm', default=True)
def xml2json(entrypoint_xml, basedir, output_file, tqdm):
    spec = readxml.parse(entrypoint_xml, basedir, enable_tqdm=tqdm)
    json.dump(spec, open(output_file, 'w+'), indent=4, sort_keys=True)
    log.info("Written to {0:s}".format(output_file))
