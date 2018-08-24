import logging
logging.basicConfig()
log = logging.getLogger(__name__)

import click
import json
import os

from . import readxml

@click.group(context_settings=dict(help_option_names=['-h', '--help']))
def pyhf():
    pass

@pyhf.command()
@click.argument('entrypoint-xml', help='The top-level XML file for the PDF definition.', type=click.Path(exists=True))
@click.option('--basedir', help='The base directory for the XML files to point relative to.', type=click.Path(exists=True), default=os.getcwd())
@click.option('--output-file', help='The location of the output json file. If not specified, prints to screen.', default=None)
@click.option('--track-progress/--hide-progress', default=True)
def xml2json(entrypoint_xml, basedir, output_file, track_progress):
    spec = readxml.parse(entrypoint_xml, basedir, track_progress=track_progress)
    if output_file is None:
        json.dumps(spec, indent=4, sort_keys=True)
    else:
        json.dump(spec, open(output_file, 'w+'), indent=4, sort_keys=True)
        log.info("Written to {0:s}".format(output_file))
