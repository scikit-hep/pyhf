"""CLI subapps to handle conversion from ROOT."""
import logging

import click
import json
import os
from pathlib import Path
import jsonpatch

log = logging.getLogger(__name__)


@click.group(name='rootio')
def cli():
    """ROOT I/O CLI group."""


@cli.command()
@click.argument('entrypoint-xml', type=click.Path(exists=True))
@click.option(
    '--basedir',
    help='The base directory for the XML files to point relative to.',
    type=click.Path(exists=True),
    default=Path.cwd(),
)
@click.option(
    '--output-file',
    help='The location of the output json file. If not specified, prints to screen.',
    default=None,
)
@click.option('--track-progress/--hide-progress', default=True)
@click.option('--validation-as-error/--validation-as-warning', default=True)
def xml2json(entrypoint_xml, basedir, output_file, track_progress, validation_as_error):
    """Entrypoint XML: The top-level XML file for the PDF definition."""
    try:
        import uproot

        assert uproot
    except ImportError:
        log.error(
            "xml2json requires uproot, please install pyhf using the "
            "xmlio extra: python -m pip install pyhf[xmlio]",
            exc_info=True,
        )
    from pyhf import readxml

    spec = readxml.parse(
        entrypoint_xml,
        basedir,
        track_progress=track_progress,
        validation_as_error=validation_as_error,
    )
    if output_file is None:
        click.echo(json.dumps(spec, indent=4, sort_keys=True))
    else:
        with open(output_file, 'w+') as out_file:
            json.dump(spec, out_file, indent=4, sort_keys=True)
        log.debug(f"Written to {output_file:s}")


@cli.command()
@click.argument('workspace', default='-')
@click.option('--output-dir', type=click.Path(exists=True), default='.')
@click.option('--specroot', default='config')
@click.option('--dataroot', default='data')
@click.option('--resultprefix', default='FitConfig')
@click.option('-p', '--patch', multiple=True)
def json2xml(workspace, output_dir, specroot, dataroot, resultprefix, patch):
    """Convert pyhf JSON back to XML + ROOT files."""
    try:
        import uproot

        assert uproot
    except ImportError:
        log.error(
            "json2xml requires uproot, please install pyhf using the "
            "xmlio extra: python -m pip install pyhf[xmlio]",
            exc_info=True,
        )
    from pyhf import writexml

    os.makedirs(output_dir, exist_ok=True)
    with click.open_file(workspace, 'r') as specstream:
        spec = json.load(specstream)
        for pfile in patch:
            patch = json.loads(click.open_file(pfile, 'r').read())
            spec = jsonpatch.JsonPatch(patch).apply(spec)
        os.makedirs(Path(output_dir).joinpath(specroot), exist_ok=True)
        os.makedirs(Path(output_dir).joinpath(dataroot), exist_ok=True)
        with click.open_file(
            Path(output_dir).joinpath(f'{resultprefix}.xml'), 'w'
        ) as outstream:
            outstream.write(
                writexml.writexml(
                    spec,
                    Path(output_dir).joinpath(specroot),
                    Path(output_dir).joinpath(dataroot),
                    resultprefix,
                ).decode('utf-8')
            )
