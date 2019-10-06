import logging

import click
import json
import os
import jsonpatch

from .utils import hypotest, EqDelimStringParamType
from .workspace import Workspace
from .version import __version__
from . import tensorlib, set_backend, optimize

logging.basicConfig()
log = logging.getLogger(__name__)

# This is only needed for Python 2/3 compatibility
def ensure_dirs(path):
    try:
        os.makedirs(path, exist_ok=True)  # lgtm [py/call/wrong-named-argument]
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
        click.echo(json.dumps(spec, indent=4, sort_keys=True))
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
@click.option('-p', '--patch', multiple=True)
def json2xml(workspace, output_dir, specroot, dataroot, resultprefix, patch):
    try:
        import uproot

        assert uproot
    except ImportError:
        log.error(
            "json2xml requires uproot, please install pyhf using the "
            "xmlio extra: pip install pyhf[xmlio] or install uproot "
            "manually: pip install uproot"
        )
    from . import writexml

    ensure_dirs(output_dir)
    with click.open_file(workspace, 'r') as specstream:
        d = json.load(specstream)
        for pfile in patch:
            patch = json.loads(click.open_file(pfile, 'r').read())
            d = jsonpatch.JsonPatch(patch).apply(d)
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
            parameter[0],
            parameter[1],
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
        (m['name'], m['config']['poi'], [p['name'] for p in m['config']['parameters']])
        for m in w.spec.get('measurements')
    ]

    maxlen_channels = max(map(len, default_model.config.channels))
    maxlen_samples = max(map(len, default_model.config.samples))
    maxlen_parameters = max(map(len, default_model.config.parameters))
    maxlen_measurements = max(map(lambda x: len(x[0]), result['measurements']))
    maxlen = max(
        [maxlen_channels, maxlen_samples, maxlen_parameters, maxlen_measurements]
    )

    # summary
    fmtStr = '{{0: >{0:d}s}}  {{1:s}}'.format(maxlen + len('Summary'))
    click.echo(fmtStr.format('     Summary     ', ''))
    click.echo(fmtStr.format('-' * 18, ''))
    fmtStr = '{{0: >{0:d}s}}  {{1:s}}'.format(maxlen)
    for key in ['channels', 'samples', 'parameters', 'modifiers']:
        click.echo(fmtStr.format(key, str(len(result[key]))))
    click.echo()

    fmtStr = '{{0: >{0:d}s}}  {{1: ^5s}}'.format(maxlen)
    click.echo(fmtStr.format('channels', 'nbins'))
    click.echo(fmtStr.format('-' * 10, '-' * 5))
    for channel, nbins in result['channels']:
        click.echo(fmtStr.format(channel, str(nbins)))
    click.echo()

    fmtStr = '{{0: >{0:d}s}}'.format(maxlen)
    click.echo(fmtStr.format('samples'))
    click.echo(fmtStr.format('-' * 10))
    for sample in result['samples']:
        click.echo(fmtStr.format(sample))
    click.echo()

    # print parameters, constraints, modifiers
    fmtStr = '{{0: >{0:d}s}}  {{1: <22s}}  {{2:s}}'.format(maxlen)
    click.echo(fmtStr.format('parameters', 'constraint', 'modifiers'))
    click.echo(fmtStr.format('-' * 10, '-' * 10, '-' * 10))
    for parname, constraint, modtypes in result['systematics']:
        click.echo(fmtStr.format(parname, constraint, ','.join(sorted(set(modtypes)))))
    click.echo()

    fmtStr = '{{0: >{0:d}s}}  {{1: ^22s}}  {{2:s}}'.format(maxlen)
    click.echo(fmtStr.format('measurement', 'poi', 'parameters'))
    click.echo(fmtStr.format('-' * 10, '-' * 10, '-' * 10))
    for measurement_name, measurement_poi, measurement_parameters in result[
        'measurements'
    ]:
        click.echo(
            fmtStr.format(
                ('(*) ' if measurement_name == default_measurement['name'] else '')
                + measurement_name,
                measurement_poi,
                ','.join(measurement_parameters)
                if measurement_parameters
                else '(none)',
            )
        )

    click.echo()

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
@click.option('--teststat', type=click.Choice(['q', 'qtilde']), default='qtilde')
@click.option('--optimizer')
@click.option('--optconf', type=EqDelimStringParamType(), multiple=True)
def cls(
    workspace, output_file, measurement, patch, testpoi, teststat, optimizer, optconf
):
    with click.open_file(workspace, 'r') as specstream:
        wspec = json.load(specstream)

    w = Workspace(wspec)

    is_qtilde = teststat == 'qtilde'

    patches = [json.loads(click.open_file(pfile, 'r').read()) for pfile in patch]
    p = w.model(
        measurement_name=measurement,
        patches=patches,
        modifier_settings={
            'normsys': {'interpcode': 'code4'},
            'histosys': {'interpcode': 'code4p'},
        },
    )

    optconf = {k: v for item in optconf for k, v in item.items()}

    # set the new optimizer
    if optimizer:
        new_optimizer = getattr(optimize, optimizer)
        set_backend(tensorlib, new_optimizer(**optconf))

    result = hypotest(testpoi, w.data(p), p, qtilde=is_qtilde, return_expected_set=True)
    result = {'CLs_obs': result[0].tolist()[0], 'CLs_exp': result[-1].ravel().tolist()}

    if output_file is None:
        click.echo(json.dumps(result, indent=4, sort_keys=True))
    else:
        with open(output_file, 'w+') as out_file:
            json.dump(result, out_file, indent=4, sort_keys=True)
        log.debug("Written to {0:s}".format(output_file))
