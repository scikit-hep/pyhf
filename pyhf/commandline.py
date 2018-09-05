import logging
logging.basicConfig()
log = logging.getLogger(__name__)

import click
import json
import os

from . import readxml
from . import writexml
from .utils import runOnePoint
from .pdf import Model

@click.group(context_settings=dict(help_option_names=['-h', '--help']))
def pyhf():
    pass

@pyhf.command()
@click.argument('entrypoint-xml', type=click.Path(exists=True))
@click.option('--basedir', help='The base directory for the XML files to point relative to.', type=click.Path(exists=True), default=os.getcwd())
@click.option('--output-file', help='The location of the output json file. If not specified, prints to screen.', default=None)
@click.option('--track-progress/--hide-progress', default=True)
def xml2json(entrypoint_xml, basedir, output_file, track_progress):
    """ Entrypoint XML: The top-level XML file for the PDF definition. """
    spec = readxml.parse(entrypoint_xml, basedir, track_progress=track_progress)
    if output_file is None:
        print(json.dumps(spec, indent=4, sort_keys=True))
    else:
        json.dump(spec, open(output_file, 'w+'), indent=4, sort_keys=True)
        log.info("Written to {0:s}".format(output_file))

@pyhf.command()
@click.argument('workspace', default = '-')
@click.argument('xmlfile', default = '-')
@click.option('--specroot', default = click.Path(exists = True))
@click.option('--dataroot', default = click.Path(exists = True))
def json2xml(workspace,xmlfile,specroot,dataroot):
    specstream = click.open_file(workspace)
    outstream = click.open_file(xmlfile,'w')
    d = json.load(specstream)

    outstream.write(writexml.writexml(d,specroot,dataroot,'').decode('utf-8'))

@pyhf.command()
@click.argument('workspace', default = '-')
@click.option('--output-file', help='The location of the output json file. If not specified, prints to screen.', default=None)
@click.option('--measurement', default=None)
@click.option('--qualify-names/--no-qualify-names', default=False)
def cls(workspace, output_file, measurement, qualify_names):
    specstream = click.open_file(workspace)
    d = json.load(specstream)
    measurements = d['toplvl']['measurements']
    measurement_names = [m['name'] for m in measurements]
    measurement_index = 0
    log.info('measurements defined:\n\t{0:s}'.format('\n\t'.join(measurement_names)))
    if measurement and measurement not in measurement_names:
        log.error('no measurement by name \'{0:s}\' exists, pick from one of the valid ones above'.format(measurement))
    else:
        if not measurement and len(measurements) > 1:
            log.warning('multiple measurements defined. Taking the first measurement.')
            measurement_index = 0
        elif measurement:
            measurement_index = measurement_names.index(measurement)

        log.info('calculating CLs for measurement {0:s}'.format(measurements[measurement_index]['name']))
        p = Model({'channels':d['channels']}, poiname=measurements[measurement_index]['config']['poi'], qualify_names=qualify_names)
        result = runOnePoint(1.0, sum((d['data'][c['name']] for c in d['channels']),[]) + p.config.auxdata, p)
        result = {'CLs_obs': result[-2].tolist()[0], 'CLs_exp': result[-1].ravel().tolist()}
        if output_file is None:
            print(json.dumps(result, indent=4, sort_keys=True))
        else:
            json.dump(result, open(output_file, 'w+'), indent=4, sort_keys=True)
            log.info("Written to {0:s}".format(output_file))
