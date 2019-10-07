import logging

import click
import json

from ..utils import validate
from ..workspace import Workspace

logging.basicConfig()
log = logging.getLogger(__name__)


@click.group(name='spec')
def cli():
    pass


@cli.command()
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
