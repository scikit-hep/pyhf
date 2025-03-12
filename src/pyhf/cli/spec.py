"""The pyhf spec CLI subcommand."""

import logging

import click
import json

from pyhf.workspace import Workspace
from pyhf import modifiers
from pyhf import parameters
from pyhf import utils
import re
from math import sqrt

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

        $ curl -sL https://raw.githubusercontent.com/scikit-hep/pyhf/main/docs/examples/json/2-bin_1-channel.json | pyhf inspect
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
    with click.open_file(workspace, "r", encoding="utf-8") as specstream:
        spec = json.load(specstream)

    ws = Workspace(spec)
    default_measurement = ws.get_measurement()

    result = {}
    result['samples'] = ws.samples
    result['channels'] = [
        (channel, ws.channel_nbins[channel]) for channel in ws.channels
    ]
    result['modifiers'] = dict(ws.modifiers)

    parset_descr = {
        parameters.paramsets.unconstrained: 'unconstrained',
        parameters.paramsets.constrained_by_normal: 'constrained_by_normal',
        parameters.paramsets.constrained_by_poisson: 'constrained_by_poisson',
    }

    model = ws.model()

    result['parameters'] = sorted(
        (parset_name, parset_descr[type(parset_spec['paramset'])])
        for parset_name, parset_spec in model.config.par_map.items()
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
    maxlen_parameters = max(map(len, [p for p, _ in result['parameters']]))
    maxlen_measurements = max(map(lambda x: len(x[0]), result['measurements']))
    maxlen = max(
        [maxlen_channels, maxlen_samples, maxlen_parameters, maxlen_measurements]
    )

    # summary
    fmtStr = '{{: >{:d}s}}  {{:s}}'.format(maxlen + len('Summary'))
    click.echo(fmtStr.format('     Summary     ', ''))
    click.echo(fmtStr.format('-' * 18, ''))
    fmtStr = f'{{0: >{maxlen:d}s}}  {{1:s}}'
    for key in ['channels', 'samples', 'parameters', 'modifiers']:
        click.echo(fmtStr.format(key, str(len(result[key]))))
    click.echo()

    fmtStr = f'{{0: >{maxlen:d}s}}  {{1: ^5s}}'
    click.echo(fmtStr.format('channels', 'nbins'))
    click.echo(fmtStr.format('-' * 10, '-' * 5))
    for channel, nbins in result['channels']:
        click.echo(fmtStr.format(channel, str(nbins)))
    click.echo()

    fmtStr = f'{{0: >{maxlen:d}s}}'
    click.echo(fmtStr.format('samples'))
    click.echo(fmtStr.format('-' * 10))
    for sample in result['samples']:
        click.echo(fmtStr.format(sample))
    click.echo()

    # print parameters, constraints, modifiers
    fmtStr = f'{{0: >{maxlen:d}s}}  {{1: <22s}}  {{2:s}}'
    click.echo(fmtStr.format('parameters', 'constraint', 'modifiers'))
    click.echo(fmtStr.format('-' * 10, '-' * 10, '-' * 10))
    for parname, constraint, modtypes in result['systematics']:
        click.echo(fmtStr.format(parname, constraint, ','.join(sorted(set(modtypes)))))
    click.echo()

    fmtStr = f'{{0: >{maxlen:d}s}}  {{1: ^22s}}  {{2:s}}'
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
                (
                    ','.join(measurement_parameters)
                    if measurement_parameters
                    else '(none)'
                ),
            )
        )

    click.echo()

    if output_file:
        with open(output_file, "w+", encoding="utf-8") as out_file:
            json.dump(result, out_file, indent=4, sort_keys=True)
        log.debug(f"Written to {output_file:s}")


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
    type=click.Choice(modifiers.histfactory_set),
)
@click.option('--measurement', default=[], multiple=True, metavar='<MEASUREMENT>...')
def prune(
    workspace, output_file, channel, sample, modifier, modifier_type, measurement
):
    """
    Prune components from the workspace.

    See :func:`pyhf.workspace.Workspace.prune` for more information.
    """
    with click.open_file(workspace, "r", encoding="utf-8") as specstream:
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
        with open(output_file, "w+", encoding="utf-8") as out_file:
            json.dump(pruned_ws, out_file, indent=4, sort_keys=True)
        log.debug(f"Written to {output_file:s}")


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
    with click.open_file(workspace, "r", encoding="utf-8") as specstream:
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
        with open(output_file, "w+", encoding="utf-8") as out_file:
            json.dump(renamed_ws, out_file, indent=4, sort_keys=True)
        log.debug(f"Written to {output_file:s}")


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
@click.option(
    '--merge-channels/--no-merge-channels',
    help='Whether or not to deeply merge channels. Can only be done with left/right outer joins.',
    default=False,
)
def combine(workspace_one, workspace_two, join, output_file, merge_channels):
    """
    Combine two workspaces into a single workspace.

    See :func:`pyhf.workspace.Workspace.combine` for more information.
    """
    with click.open_file(workspace_one, "r", encoding="utf-8") as specstream:
        spec_one = json.load(specstream)

    with click.open_file(workspace_two, "r", encoding="utf-8") as specstream:
        spec_two = json.load(specstream)

    ws_one = Workspace(spec_one)
    ws_two = Workspace(spec_two)
    combined_ws = Workspace.combine(
        ws_one, ws_two, join=join, merge_channels=merge_channels
    )

    if output_file is None:
        click.echo(json.dumps(combined_ws, indent=4, sort_keys=True))
    else:
        with open(output_file, "w+", encoding="utf-8") as out_file:
            json.dump(combined_ws, out_file, indent=4, sort_keys=True)
        log.debug(f"Written to {output_file:s}")


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

        $ curl -sL https://raw.githubusercontent.com/scikit-hep/pyhf/main/docs/examples/json/2-bin_1-channel.json | pyhf digest
        sha256:dad8822af55205d60152cbe4303929042dbd9d4839012e055e7c6b6459d68d73
    """
    with click.open_file(workspace, "r", encoding="utf-8") as specstream:
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

        $ curl -sL https://raw.githubusercontent.com/scikit-hep/pyhf/main/docs/examples/json/2-bin_1-channel.json | pyhf sort | jq '.' | md5
        8be5186ec249d2704e14dd29ef05ffb0

    .. code-block:: shell

        $ curl -sL https://raw.githubusercontent.com/scikit-hep/pyhf/main/docs/examples/json/2-bin_1-channel.json | jq -S '.channels|=sort_by(.name)|.channels[].samples|=sort_by(.name)|.channels[].samples[].modifiers|=sort_by(.name,.type)|.observations|=sort_by(.name)' | md5
        8be5186ec249d2704e14dd29ef05ffb0


    """
    with click.open_file(workspace, "r", encoding="utf-8") as specstream:
        spec = json.load(specstream)

    workspace = Workspace(spec)
    sorted_ws = Workspace.sorted(workspace)

    if output_file is None:
        click.echo(json.dumps(sorted_ws, indent=4, sort_keys=True))
    else:
        with open(output_file, "w+", encoding="utf-8") as out_file:
            json.dump(sorted_ws, out_file, indent=4, sort_keys=True)
        log.debug(f"Written to {output_file}")


def stripname(name, stripcomponents):
    while True:
        anymatch = False
        for c in stripcomponents:
            if name.endswith(c):
                name = name[: -len(c)].strip("_")
                anymatch = True
                break
            if name.startswith(c):
                name = name[len(c) :].strip("_")
                anymatch = True
                break
            if c in name:
                name = name.replace("_" + c + "_", "_")
                anymatch = True
                break
        if not anymatch:
            return name


@cli.command()
@click.argument('workspace', default='-')
@click.option(
    '--output-file',
    help='The location of the output json file. If not specified, prints to screen.',
    default=None,
)
@click.option("--poi", default=None, help="Parameter of Interest")
@click.option("--data", default="obsData", help="Name of the dataset")
@click.option("--fix-other-pois", is_flag=True, help="Fix other POIs")
@click.option(
    "--nf-blacklist",
    multiple=True,
    default=["binWidth_.*", "one"],
    help="Blacklist for normfactors",
)
@click.option(
    "--sys-blacklist", multiple=True, default=["zero"], help="Blacklist for systematics"
)
@click.option(
    "--strip-name-components",
    multiple=True,
    default=["model"],
    help="Components to strip from names",
)
@click.option(
    "--defactorize", is_flag=True, help="Merge OverallSys and Histosys of the same name"
)
def hs3_to_hifa(
    workspace,
    output_file,
    poi,
    data,
    fix_other_pois,
    nf_blacklist,
    sys_blacklist,
    strip_name_components,
    defactorize,
):
    """
    Convert the HS3 workspace to a HiFa JSON workspace for use with pyhf.

    Taken from: https://gitlab.cern.ch/cburgard/RooFitUtils/-/blob/bbb079c1f597b138e2b638b0d98cff3835596240/scripts/json-roofit2pyhf.py

    Example:

    .. code-block:: shell

        $ curl -sL https://raw.githubusercontent.com/root-project/root/12b7ffe/tutorials/roofit/roofit/rf515_hfJSON.json | pyhf sort | jq '.' | md5
        8be5186ec249d2704e14dd29ef05ffb0

    """
    with click.open_file(workspace, "r", encoding="utf-8") as specstream:
        spec = json.load(specstream)

    variables = spec.get("variables", {})
    bounds = {}
    for domain in spec.get("domains", []):
        for parameter in domain["axes"]:
            bounds[parameter["name"]] = [parameter["min"], parameter["max"]]

    # setup the main structure
    pois = set()
    for analysis in spec["analyses"]:
        for poi in analysis["parameters_of_interest"]:
            pois.add(poi)
    measurement = {"name": "meas", "config": {"parameters": []}}
    if poi:
        measurement["config"]["poi"] = poi
    elif len(pois) > 0:
        measurement["config"]["poi"] = next(iter(pois))
    output_json = {
        "channels": [],
        "measurements": [measurement],
        "observations": [],
        "version": "1.0.0",
    }

    # some bookkeeping
    nps = set()
    nfs = set()
    channelnames = []

    # define observations / data
    this_data = {}
    for key, channel in this_data.items():
        if not isinstance(channel, dict):
            continue
        channelname = stripname(key, strip_name_components)
        channelnames.append(channelname)
        if "counts" in channel.keys():
            output_json["observations"].append(
                {"data": channel["counts"], "name": channelname}
            )
        else:
            output_json["observations"].append(
                {"data": channel["weights"], "name": channelname}
            )

    observations = []
    parameters = {}

    # define model /pdf
    for pdf in sorted(spec.get("distributions", []), key=lambda x: x["name"]):
        if pdf["type"] != "histfactory_dist":
            continue
        if "name" in pdf.keys():
            key = pdf["name"]
        channelname = stripname(key, strip_name_components)
        for c in channelnames:
            if c.startswith(channelname):
                channelname = c

        for any_data in spec["data"]:
            if data in any_data["name"] and channelname in any_data["name"]:
                observations.append({"data": any_data["contents"], "name": channelname})

        out_channel = {"name": channelname, "samples": []}
        output_json["channels"].append(out_channel)

        if "samples" not in pdf.keys():
            print("workspace is no histfactory workspace")
            exit(1)

        sum_values = None
        sum_errors2 = None

        for sample in pdf["samples"]:
            if "data" not in sample.keys():
                print("workspace no histfactory workspace")
                exit(1)
            has_staterror = False
            for modifier in sample["modifiers"]:
                if modifier["type"] == "staterror":
                    has_staterror = True

            if has_staterror:
                values = sample["data"]["contents"]
                if sum_values:
                    for i in range(len(sum_values)):
                        sum_values[i] += values[i]
                else:
                    sum_values = [v for v in values]

                errors = sample["data"]["errors"]
                if sum_errors2:
                    for i in range(len(sum_errors2)):
                        sum_errors2[i] += errors[i] * errors[i]
                else:
                    sum_errors2 = [e * e for e in errors]

        for sample in sorted(pdf["samples"], key=lambda x: x["name"]):
            values = sample["data"]["contents"]
            bins = len(values)

            out_sample = {
                "name": stripname(sample["name"], strip_name_components),
                "modifiers": [],
            }
            out_channel["samples"].append(out_sample)
            out_sample["data"] = values

            modifiers = out_sample["modifiers"]
            for modifier in sorted(sample["modifiers"], key=lambda x: x["name"]):
                if modifier.get("constraint", None) == "Const":
                    continue
                if modifier["name"] == "Lumi":
                    modifiers.append({"name": "lumi", "type": "lumi", "data": None})
                    nps.add("lumi")
                    parameters["lumi"] = {
                        "fixed": False,
                        "inits": [1.0],
                        "auxdata": [1.0],
                        **(
                            {"bounds": [bounds[modifier["name"]]]}
                            if modifier["name"] in bounds
                            else {}
                        ),
                    }
                elif modifier["type"] == "staterror":
                    parname = "staterror_" + channelname
                    modifiers.append(
                        {
                            "name": parname,
                            "type": "staterror",
                            "data": [
                                sqrt(sum_errors2[i]) / sum_values[i] * values[i]
                                for i in range(bins)
                            ],
                        }
                    )
                    nps.add(parname)
                    parameters[parname] = {
                        "fixed": False,
                        "auxdata": [1.0 for i in range(bins)],
                        "inits": [1.0 for i in range(bins)],
                        "bounds": [[-5.0, 5.0] for i in range(bins)],
                    }
                elif modifier["type"] == "normfactor":
                    modifiers.append(
                        {
                            "name": modifier["name"],
                            "type": "normfactor",
                            "data": None,
                        }
                    )
                    nfs.add(modifier["name"])
                    parameters[modifier["name"]] = {
                        "fixed": False,
                        "inits": [1.0],
                        **(
                            {"bounds": [bounds[modifier["name"]]]}
                            if modifier["name"] in bounds
                            else {}
                        ),
                    }
                elif modifier["type"] == "normsys":
                    modifiers.append(
                        {
                            "name": modifier["name"],
                            "type": "normsys",
                            "data": modifier["data"],
                        }
                    )
                    nps.add(modifier["name"])
                    parameters[modifier["name"]] = {
                        "fixed": False,
                        "auxdata": [0.0],
                        "inits": [0.0],
                        **(
                            {"bounds": [bounds[modifier["name"]]]}
                            if modifier["name"] in bounds
                            else {}
                        ),
                    }
                elif modifier["type"] == "shapesys":
                    parname = modifier["name"]
                    nps.add(parname)
                    modifiers.append(
                        {
                            "name": parname,
                            "type": "shapesys",
                            "data": modifier["data"]["vals"],
                        }
                    )
                    parameters[parname] = {
                        "fixed": False,
                        "auxdata": [1.0 for i in range(bins)],
                        "inits": [0.0 for i in range(bins)],
                        "bounds": [[-5.0, 5.0] for i in range(bins)],
                    }
                    if bins == 9:
                        print(parname)
                elif modifier["type"] == "histosys":
                    modifiers.append(
                        {
                            "name": modifier["name"],
                            "type": "histosys",
                            "data": {
                                "hi_data": modifier["data"]["hi"]["contents"],
                                "lo_data": modifier["data"]["lo"]["contents"],
                            },
                        }
                    )
                    nps.add(modifier["name"])
                    parameters[modifier["name"]] = {
                        "fixed": False,
                        "auxdata": [0.0],
                        "inits": [0.0],
                        **(
                            {"bounds": [bounds[modifier["name"]]]}
                            if modifier["name"] in bounds
                            else {}
                        ),
                    }
                else:
                    print(
                        f"workspace contains unknown modifier type: {modifier['type']}"
                    )
                    exit(1)

    # define parameters
    for par in nps:
        parameters[par]["name"] = par
        measurement["config"]["parameters"].append(parameters[par])
    for par in nfs:
        if any([re.match(b, par) for b in nf_blacklist]):
            continue
        parameters[par]["name"] = par
        measurement["config"]["parameters"].append(parameters[par])

    # some post-processing
    for p in measurement["config"]["parameters"]:
        pname = p["name"]
        if pname in variables.keys():
            if "const" in variables[pname].keys():
                p["fixed"] = variables[pname]["const"]
        if fix_other_pois:
            for this_poi in pois:
                if this_poi == poi:
                    continue
                if pname == this_poi:
                    p["fixed"] = True

    # write observations
    output_json["observations"] = observations

    if output_file is None:
        click.echo(json.dumps(output_json, indent=4, sort_keys=True))
    else:
        with open(output_file, "w+", encoding="utf-8") as out_file:
            json.dump(output_json, out_file, indent=4, sort_keys=True)
        log.debug(f"Written to {output_file}")
