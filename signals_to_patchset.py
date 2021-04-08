#!/usr/bin/env python
import pyhf
import click
import json
import parse
import sys
import pathlib
import jsonpatch


@parse.with_pattern(r'\d+(?:[p\.]\d+)?')
def parse_number(text):
    return float(text.replace('p', '.'))


@click.command()
@click.argument("signals", nargs=-1, type=click.File("r"))
@click.option(
    '-b',
    '--bkg-only',
    help="Background-only JSON file",
    required=True,
    type=click.File("r"),
)
@click.option(
    "-p",
    "--pattern",
    help="Pattern of signal filenames to extract information from",
    default="{x:number}_{y:number}",
)
@click.option('-i', '--analysis-id', default=None, help="Analysis ID if it exists")
@click.option(
    '-a',
    '--algorithm',
    'algorithms',
    default=['sha256'],
    help='Digest algorithms to use',
    multiple=True,
)
# @click.option(
#    '-v',
#    '--version',
#    default="1.0.0",
#    help="patchset version",
# )
@click.option(
    '-r',
    '--references',
    help="string containing json-like dictionary containing references to the analysis, e.g. \"{'hepdata':'ins1234567'}\"",
    required=True,
    type=str,
)
@click.option(
    '-d',
    '--description',
    default="signal patchset",
    help="Description of patchset file",
)
@click.option(
    "--output-file",
    help="The location of the output json file. If not specified, prints to screen.",
    default=None,
)
def main(
    signals,
    bkg_only,
    pattern,
    algorithms,
    description,
    analysis_id,
    output_file,
    references,
):
    patchset = {
        'metadata': {
            'description': description,
        },
        'patches': [],
    }

    # add in analysis id if specified
    if analysis_id:
        patchset['metadata']['analysis_id'] = analysis_id
    # add in the digest for background-only
    bkg_only_workspace = json.load(bkg_only)
    patchset['metadata']['digests'] = {
        algorithm: pyhf.utils.digest(bkg_only_workspace, algorithm=algorithm)
        for algorithm in algorithms
    }
    click.echo("Background-only digests:")
    click.echo(
        '\t'
        + json.dumps(patchset['metadata']['digests'], indent=4, sort_keys=True).replace(
            '\n', '\n\t'
        )
    )
    p = parse.compile(pattern, dict(number=parse_number))
    labels = p._named_fields
    if len(labels) == 0:
        click.echo("You didn't specify any named labels...")
        sys.exit(1)
    click.echo(f"You specified {len(labels)} labels: {labels}.")
    patchset['metadata']['labels'] = labels
    patchset['metadata']['references'] = json.loads(references)
    patchset['version'] = "1.0.0"

    click.echo(f"Making patches for {len(signals)} signals.")
    for signal in signals:
        signal_name = pathlib.Path(signal.name).stem
        r = p.search(signal_name)
        if r is None:
            click.echo(f"Filename parse failure for: {signal.name}")
            sys.exit(1)
        values = [r.named[label] for label in labels]

        patch = jsonpatch.make_patch(bkg_only_workspace, json.load(signal))
        if not patch.patch:
            click.echo(f"Patch failure for: {signal.name}")
            sys.exit(1)
        patchset['patches'].append(
            {'metadata': {'name': signal_name, 'values': values}, 'patch': patch.patch}
        )

    click.echo("Done. Validating patchset structure against schema.")
    pyhf.utils.validate(patchset, 'patchset.json')
    click.echo(
        f"Validated. {'Writing to file' if output_file else 'Printing to screen'}."
    )
    if output_file is None:
        click.echo(json.dumps(patchset, indent=4, sort_keys=True))
    else:
        with open(output_file, "w+") as out_file:
            json.dump(patchset, out_file, indent=4, sort_keys=True)
        click.echo(f"Written to {output_file:s}")


if __name__ == "__main__":
    main()
