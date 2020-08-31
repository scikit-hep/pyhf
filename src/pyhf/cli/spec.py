"""The pyhf spec CLI subcommand."""
import logging

import click
import json

from ..workspace import Workspace
from .. import modifiers
from .. import utils

log = logging.getLogger(__name__)


@click.group(name='spec')
def cli():
    """Spec CLI group."""


@cli.command()
@click.argument('workspace', default='-')
@click.option(
    '--output-file',
    help='The location of the output json file. If not specified, prints to screen.',
    default=None,
)
@click.option('--measurement', default=None)
def inspect(workspace, output_file, measurement):
    """
    Inspect a pyhf JSON document.

    Example:

    .. code-block:: shell

        $ curl -sL https://raw.githubusercontent.com/scikit-hep/pyhf/master/docs/examples/json/2-bin_1-channel.json | pyhf inspect
                  Summary
            ------------------
               channels  1
                samples  2
             parameters  2
              modifiers  2

               channels  nbins
             ----------  -----
          singlechannel    2

                samples
             ----------
             background
                 signal

             parameters  constraint              modifiers
             ----------  ----------              ----------
                     mu  unconstrained           normfactor
        uncorr_bkguncrt  constrained_by_poisson  shapesys

            measurement           poi            parameters
             ----------        ----------        ----------
        (*) Measurement            mu            (none)

    """
    with click.open_file(workspace, 'r') as specstream:
        spec = json.load(specstream)

    ws = Workspace(spec)
    default_measurement = ws.get_measurement()

    result = {}
    result['samples'] = ws.samples
    result['channels'] = [
        (channel, ws.channel_nbins[channel]) for channel in ws.channels
    ]
    result['modifiers'] = dict(ws.modifiers)

    result['parameters'] = sorted(
        (
            parname,
            modifiers.registry[result['modifiers'][parname]]
            .required_parset([], [])['paramset_type']
            .__name__,
        )
        for parname in ws.parameters
    )
    result['systematics'] = [
        (
            parameter[0],
            parameter[1],
            [modifier[1] for modifier in ws.modifiers if modifier[0] == parameter[0]],
        )
        for parameter in result['parameters']
    ]

    result['measurements'] = [
        (m['name'], m['config']['poi'], [p['name'] for p in m['config']['parameters']])
        for m in ws.get('measurements')
    ]

    maxlen_channels = max(map(len, ws.channels))
    maxlen_samples = max(map(len, ws.samples))
    maxlen_parameters = max(map(len, ws.parameters))
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


@cli.command()
@click.argument('workspace', default='-')
@click.option(
    '--output-file',
    help='The location of the output json file. If not specified, prints to screen.',
    default=None,
)
@click.option('-c', '--channel', default=[], multiple=True, metavar='<CHANNEL>...')
@click.option('-s', '--sample', default=[], multiple=True, metavar='<SAMPLE>...')
@click.option('-m', '--modifier', default=[], multiple=True, metavar='<MODIFIER>...')
@click.option(
    '-t',
    '--modifier-type',
    default=[],
    multiple=True,
    type=click.Choice(modifiers.uncombined.keys()),
)
@click.option('--measurement', default=[], multiple=True, metavar='<MEASUREMENT>...')
def prune(
    workspace, output_file, channel, sample, modifier, modifier_type, measurement
):
    """
    Prune components from the workspace.

    See :func:`pyhf.workspace.Workspace.prune` for more information.
    """
    with click.open_file(workspace, 'r') as specstream:
        spec = json.load(specstream)

    ws = Workspace(spec)
    pruned_ws = ws.prune(
        channels=channel,
        samples=sample,
        modifiers=modifier,
        modifier_types=modifier_type,
        measurements=measurement,
    )

    if output_file is None:
        click.echo(json.dumps(pruned_ws, indent=4, sort_keys=True))
    else:
        with open(output_file, 'w+') as out_file:
            json.dump(pruned_ws, out_file, indent=4, sort_keys=True)
        log.debug("Written to {0:s}".format(output_file))


@cli.command()
@click.argument('workspace', default='-')
@click.option(
    '--output-file',
    help='The location of the output json file. If not specified, prints to screen.',
    default=None,
)
@click.option(
    '-c',
    '--channel',
    default=[],
    multiple=True,
    type=click.Tuple([str, str]),
    metavar='<PATTERN> <REPLACE>...',
)
@click.option(
    '-s',
    '--sample',
    default=[],
    multiple=True,
    type=click.Tuple([str, str]),
    metavar='<PATTERN> <REPLACE>...',
)
@click.option(
    '-m',
    '--modifier',
    default=[],
    multiple=True,
    type=click.Tuple([str, str]),
    metavar='<PATTERN> <REPLACE>...',
)
@click.option(
    '--measurement',
    default=[],
    multiple=True,
    type=click.Tuple([str, str]),
    metavar='<PATTERN> <REPLACE>...',
)
def rename(workspace, output_file, channel, sample, modifier, measurement):
    """
    Rename components of the workspace.

    See :func:`pyhf.workspace.Workspace.rename` for more information.
    """
    with click.open_file(workspace, 'r') as specstream:
        spec = json.load(specstream)

    ws = Workspace(spec)
    renamed_ws = ws.rename(
        channels=dict(channel),
        samples=dict(sample),
        modifiers=dict(modifier),
        measurements=dict(measurement),
    )

    if output_file is None:
        click.echo(json.dumps(renamed_ws, indent=4, sort_keys=True))
    else:
        with open(output_file, 'w+') as out_file:
            json.dump(renamed_ws, out_file, indent=4, sort_keys=True)
        log.debug("Written to {0:s}".format(output_file))


@cli.command()
@click.argument('workspace-one', default='-')
@click.argument('workspace-two', default='-')
@click.option(
    '-j',
    '--join',
    default='none',
    type=click.Choice(Workspace.valid_joins),
    help='The join operation to apply when combining the two workspaces.',
)
@click.option(
    '--output-file',
    help='The location of the output json file. If not specified, prints to screen.',
    default=None,
)
def combine(workspace_one, workspace_two, join, output_file):
    """
    Combine two workspaces into a single workspace.

    See :func:`pyhf.workspace.Workspace.combine` for more information.
    """
    with click.open_file(workspace_one, 'r') as specstream:
        spec_one = json.load(specstream)

    with click.open_file(workspace_two, 'r') as specstream:
        spec_two = json.load(specstream)

    ws_one = Workspace(spec_one)
    ws_two = Workspace(spec_two)
    combined_ws = Workspace.combine(ws_one, ws_two, join=join)

    if output_file is None:
        click.echo(json.dumps(combined_ws, indent=4, sort_keys=True))
    else:
        with open(output_file, 'w+') as out_file:
            json.dump(combined_ws, out_file, indent=4, sort_keys=True)
        log.debug("Written to {0:s}".format(output_file))


@cli.command()
@click.argument('workspace', default='-')
@click.option(
    '-a',
    '--algorithm',
    default=['sha256'],
    multiple=True,
    help='The hashing algorithm used to compute the workspace digest.',
)
@click.option(
    '-j/-p',
    '--json/--plaintext',
    'output_json',
    help='Output the hash values as a JSON dictionary or plaintext strings',
)
def digest(workspace, algorithm, output_json):
    """
    Use hashing algorithm to calculate the workspace digest.

    Returns:
        digests (:obj:`dict`): A mapping of the hashing algorithms used to the computed digest for the workspace.

    Example:

    .. code-block:: shell

        $ curl -sL https://raw.githubusercontent.com/scikit-hep/pyhf/master/docs/examples/json/2-bin_1-channel.json | pyhf digest
        sha256:dad8822af55205d60152cbe4303929042dbd9d4839012e055e7c6b6459d68d73
    """
    with click.open_file(workspace, 'r') as specstream:
        spec = json.load(specstream)

    workspace = Workspace(spec)

    digests = {
        hash_alg: utils.digest(workspace, algorithm=hash_alg) for hash_alg in algorithm
    }

    if output_json:
        output = json.dumps(digests, indent=4, sort_keys=True)
    else:
        output = '\n'.join(
            f"{hash_alg}:{digest}" for hash_alg, digest in digests.items()
        )

    click.echo(output)


@cli.command()
@click.argument('workspace', default='-')
@click.option(
    '--output-file',
    help='The location of the output json file. If not specified, prints to screen.',
    default=None,
)
def sort(workspace, output_file):
    """
    Sort the workspace.

    See :func:`pyhf.workspace.Workspace.sorted` for more information.

    Example:

    .. code-block:: shell

        $ curl -sL https://raw.githubusercontent.com/scikit-hep/pyhf/master/docs/examples/json/2-bin_1-channel.json | pyhf sort | jq '.' | md5
        8be5186ec249d2704e14dd29ef05ffb0

    .. code-block:: shell

        $ curl -sL https://raw.githubusercontent.com/scikit-hep/pyhf/master/docs/examples/json/2-bin_1-channel.json | jq -S '.channels|=sort_by(.name)|.channels[].samples|=sort_by(.name)|.channels[].samples[].modifiers|=sort_by(.name,.type)|.observations|=sort_by(.name)' | md5
        8be5186ec249d2704e14dd29ef05ffb0


    """
    with click.open_file(workspace, 'r') as specstream:
        spec = json.load(specstream)

    workspace = Workspace(spec)
    sorted_ws = Workspace.sorted(workspace)

    if output_file is None:
        click.echo(json.dumps(sorted_ws, indent=4, sort_keys=True))
    else:
        with open(output_file, 'w+') as out_file:
            json.dump(sorted_ws, out_file, indent=4, sort_keys=True)
        log.debug(f"Written to {output_file}")
