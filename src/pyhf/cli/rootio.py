import logging

import click
import json
import os
import jsonpatch

logging.basicConfig()
log = logging.getLogger(__name__)


@click.group(name='rootio')
def cli():
    pass


@cli.command()
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
            "xmlio extra: pip install pyhf[xmlio]"
        )
    from .. import readxml

    spec = readxml.parse(entrypoint_xml, basedir, track_progress=track_progress)
    if output_file is None:
        click.echo(json.dumps(spec, indent=4, sort_keys=True))
    else:
        with open(output_file, 'w+') as out_file:
            json.dump(spec, out_file, indent=4, sort_keys=True)
        log.debug("Written to {0:s}".format(output_file))


@cli.command()
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
            "xmlio extra: pip install pyhf[xmlio]"
        )
    from .. import writexml

    os.makedirs(output_dir, exist_ok=True)
    with click.open_file(workspace, 'r') as specstream:
        spec = json.load(specstream)
        for pfile in patch:
            patch = json.loads(click.open_file(pfile, 'r').read())
            spec = jsonpatch.JsonPatch(patch).apply(spec)
        os.makedirs(os.path.join(output_dir, specroot), exist_ok=True)
        os.makedirs(os.path.join(output_dir, dataroot), exist_ok=True)
        with click.open_file(
            os.path.join(output_dir, '{0:s}.xml'.format(resultprefix)), 'w'
        ) as outstream:
            outstream.write(
                writexml.writexml(
                    spec,
                    os.path.join(output_dir, specroot),
                    os.path.join(output_dir, dataroot),
                    resultprefix,
                ).decode('utf-8')
            )
