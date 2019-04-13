import logging

import click
import json
import os

from . import writexml
from .utils import hypotest
from .pdf import Workspace
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
            "xmlio extra: pip install pyhf[xmlio] or install uproot "
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


@pyhf.command()
@click.argument('workspace', default='-')
@click.option('--output-dir', type=click.Path(exists=True), default='.')
@click.option('--specroot', default='config')
@click.option('--dataroot', default='data')
@click.option('--resultprefix', default='FitConfig')
def json2xml(workspace, output_dir, specroot, dataroot, resultprefix):
    try:
        import uproot

        assert uproot
    except ImportError:
        log.error(
            "json2xml requires uproot, please install pyhf using the "
            "xmlio extra: pip install pyhf[xmlio] or install uproot "
            "manually: pip install uproot"
        )

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


@pyhf.command()
@click.argument('workspace', default='-')
@click.option(
    '--output-file',
    help='The location of the output json file. If not specified, prints to screen.',
    default=None,
)
@click.option('--measurement', default=None)
def inspect(workspace, output_file, measurement):
    with click.open_file(workspace, 'r') as specstream:
        wspec = json.load(specstream)

    w = Workspace(wspec)
    default_measurement = w.get_measurement()
    default_model = w.model(measurement_name=default_measurement['name'])

    result = {}
    result['samples'] = default_model.config.samples
    result['channels'] = [
        (channel, default_model.config.channel_nbins[channel])
        for channel in default_model.config.channels
    ]
    result['parameters'] = sorted(
        (parname, default_model.config.par_map[parname]['paramset'].__class__.__name__)
        for parname in default_model.config.parameters
    )
    result['systematics'] = [
        (
            *parameter,
            [
                modifier[1]
                for modifier in default_model.config.modifiers
                if modifier[0] == parameter[0]
            ],
        )
        for parameter in result['parameters']
    ]

    result['modifiers'] = default_model.config.modifiers

    result['measurements'] = [
        (
            ('(*) ' if measurement['name'] == default_measurement['name'] else '')
            + measurement['name'],
            measurement['config']['poi'],
            [p['name'] for p in measurement['config']['parameters']],
        )
        for measurement in w.spec.get('measurements')
    ]

    maxlen_channels = max(map(len, default_model.config.channels))
    maxlen_samples = max(map(len, default_model.config.samples))
    maxlen_parameters = max(map(len, default_model.config.parameters))
    maxlen_measurements = max(map(lambda x: len(x[0]), result['measurements']))
    maxlen = max(
        [maxlen_channels, maxlen_samples, maxlen_parameters, maxlen_measurements]
    )

    # summary statistics
    fmtStr = '{{0: >{0:d}s}}  {{1:s}}'.format(maxlen + len('Summary'))
    print(fmtStr.format('Summary Statistics', ''))
    print(fmtStr.format('-' * 18, ''))
    fmtStr = '{{0: >{0:d}s}}  {{1:s}}'.format(maxlen)
    for key in ['channels', 'samples', 'parameters', 'modifiers']:
        print(fmtStr.format(key, str(len(result[key]))))
    print()

    fmtStr = '{{0: >{0:d}s}}  {{1: ^5s}}'.format(maxlen)
    print(fmtStr.format('channels', 'nbins'))
    print(fmtStr.format('-' * 10, '-' * 5))
    for channel, nbins in result['channels']:
        print(fmtStr.format(channel, str(nbins)))
    print()

    fmtStr = '{{0: >{0:d}s}}'.format(maxlen)
    print(fmtStr.format('samples'))
    print(fmtStr.format('-' * 10))
    for sample in result['samples']:
        print(fmtStr.format(sample))
    print()

    # print parameters, constraints, modifiers
    fmtStr = '{{0: >{0:d}s}}  {{1: <22s}}  {{2:s}}'.format(maxlen)
    print(fmtStr.format('parameters', 'constraint', 'modifiers'))
    print(fmtStr.format('-' * 10, '-' * 10, '-' * 10))
    for parname, constraint, modtypes in result['systematics']:
        print(fmtStr.format(parname, constraint, ','.join(sorted(set(modtypes)))))
    print()

    fmtStr = '{{0: >{0:d}s}}  {{1: ^22s}}  {{2:s}}'.format(maxlen)
    print(fmtStr.format('measurement', 'poi', 'parameters'))
    print(fmtStr.format('-' * 10, '-' * 10, '-' * 10))
    for measurement_name, measurement_poi, measurement_parameters in result[
        'measurements'
    ]:
        print(
            fmtStr.format(
                measurement_name,
                measurement_poi,
                ','.join(measurement_parameters)
                if measurement_parameters
                else '(none)',
            )
        )
    print()

    if output_file:
        with open(output_file, 'w+') as out_file:
            json.dump(result, out_file, indent=4, sort_keys=True)
        log.debug("Written to {0:s}".format(output_file))


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
        wspec = json.load(specstream)

    w = Workspace(wspec)

    patches = [json.loads(click.open_file(pfile, 'r').read()) for pfile in patch]
    p = w.model(measurement_name=measurement, patches=patches)
    result = hypotest(testpoi, w.data(p), p, return_expected_set=True)
    result = {'CLs_obs': result[0].tolist()[0], 'CLs_exp': result[-1].ravel().tolist()}
    if output_file is None:
        print(json.dumps(result, indent=4, sort_keys=True))
    else:
        with open(output_file, 'w+') as out_file:
            json.dump(result, out_file, indent=4, sort_keys=True)
        log.debug("Written to {0:s}".format(output_file))
