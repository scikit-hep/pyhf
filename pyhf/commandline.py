import logging

import click
import json
import os
import jsonpatch
import sys

from . import writexml
from .utils import hypotest
from .pdf import Model
from .version import __version__

logging.basicConfig()
log = logging.getLogger(__name__)

# This is only needed for Python 2/3 compatibility
def ensure_dirs(path):
    try:
        os.makedirs(path, exist_ok=True)
    except TypeError:
        if not os.path.exists(path):
            os.makedirs(path)


@click.group(context_settings=dict(help_option_names=['-h', '--help']))
@click.version_option(version=__version__)
def pyhf():
    pass


@pyhf.command()
@click.argument('entrypoint-xml', type=click.Path(exists=True))
@click.option(
    '--basedir',
    help='The base directory for the XML files to point relative to.',
    type=click.Path(exists=True),
    default=os.getcwd(),
)
@click.option(
    '--output-file',
    help='The location of the output json file. If not specified, prints to screen.',
    default=None,
)
@click.option('--track-progress/--hide-progress', default=True)
def xml2json(entrypoint_xml, basedir, output_file, track_progress):
    """ Entrypoint XML: The top-level XML file for the PDF definition. """
    try:
        import uproot

        assert uproot
    except ImportError:
        log.error(
            "xml2json requires uproot, please install pyhf using the "
            "xmlimport extra: pip install pyhf[xmlimport] or install uproot "
            "manually: pip install uproot"
        )
    from . import readxml

    spec = readxml.parse(entrypoint_xml, basedir, track_progress=track_progress)
    if output_file is None:
        print(json.dumps(spec, indent=4, sort_keys=True))
    else:
        with open(output_file, 'w+') as out_file:
            json.dump(spec, out_file, indent=4, sort_keys=True)
        log.debug("Written to {0:s}".format(output_file))
    sys.exit(0)


@pyhf.command()
@click.argument('workspace', default='-')
@click.option('--output-dir', type=click.Path(exists=True), default='.')
@click.option('--specroot', default='config')
@click.option('--dataroot', default='data')
@click.option('--resultprefix', default='FitConfig')
def json2xml(workspace, output_dir, specroot, dataroot, resultprefix):
    ensure_dirs(output_dir)
    with click.open_file(workspace, 'r') as specstream:
        d = json.load(specstream)
        ensure_dirs(os.path.join(output_dir, specroot))
        ensure_dirs(os.path.join(output_dir, dataroot))
        with click.open_file(
            os.path.join(output_dir, '{0:s}.xml'.format(resultprefix)), 'w'
        ) as outstream:
            outstream.write(
                writexml.writexml(
                    d,
                    os.path.join(output_dir, specroot),
                    os.path.join(output_dir, dataroot),
                    resultprefix,
                ).decode('utf-8')
            )

    sys.exit(0)


@pyhf.command()
@click.argument('workspace', default='-')
@click.option(
    '--output-file',
    help='The location of the output json file. If not specified, prints to screen.',
    default=None,
)
@click.option('--measurement', default=None)
@click.option('-p', '--patch', multiple=True)
@click.option('--testpoi', default=1.0)
def cls(workspace, output_file, measurement, patch, testpoi):
    with click.open_file(workspace, 'r') as specstream:
        d = json.load(specstream)
    measurements = d['toplvl']['measurements']
    measurement_names = [m['name'] for m in measurements]
    measurement_index = 0

    log.debug('measurements defined:\n\t{0:s}'.format('\n\t'.join(measurement_names)))
    if measurement and measurement not in measurement_names:
        log.error(
            'no measurement by name \'{0:s}\' exists, pick from one of the valid ones above'.format(
                measurement
            )
        )
        sys.exit(1)
    else:
        if not measurement and len(measurements) > 1:
            log.warning('multiple measurements defined. Taking the first measurement.')
            measurement_index = 0
        elif measurement:
            measurement_index = measurement_names.index(measurement)

        log.debug(
            'calculating CLs for measurement {0:s}'.format(
                measurements[measurement_index]['name']
            )
        )
        spec = {
            'channels': d['channels'],
            'parameters': d['toplvl']['measurements'][measurement_index]['config'].get(
                'parameters', []
            ),
        }

        for p in patch:
            with click.open_file(p, 'r') as read_file:
                p = jsonpatch.JsonPatch(json.loads(read_file.read()))
            spec = p.apply(spec)
        p = Model(spec, poiname=measurements[measurement_index]['config']['poi'])
        observed = sum((d['data'][c] for c in p.config.channels), []) + p.config.auxdata
        result = hypotest(testpoi, observed, p, return_expected_set=True)
        result = {
            'CLs_obs': result[0].tolist()[0],
            'CLs_exp': result[-1].ravel().tolist(),
        }
        if output_file is None:
            print(json.dumps(result, indent=4, sort_keys=True))
        else:
            with open(output_file, 'w+') as out_file:
                json.dump(result, out_file, indent=4, sort_keys=True)
            log.debug("Written to {0:s}".format(output_file))
        sys.exit(0)
