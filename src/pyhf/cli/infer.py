"""The inference CLI group."""
import logging

import click
import json

from ..utils import EqDelimStringParamType
from ..infer import hypotest
from ..workspace import Workspace
from .. import get_backend, set_backend, optimize

log = logging.getLogger(__name__)


@click.group(name='infer')
def cli():
    """Infererence CLI group."""


@cli.command()
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
@click.option(
    '--backend',
    type=click.Choice(['numpy', 'pytorch', 'tensorflow', 'jax', 'np', 'torch', 'tf']),
    help='The tensor backend used for the calculation.',
    default='numpy',
)
@click.option('--optimizer')
@click.option('--optconf', type=EqDelimStringParamType(), multiple=True)
def cls(
    workspace,
    output_file,
    measurement,
    patch,
    testpoi,
    teststat,
    backend,
    optimizer,
    optconf,
):
    """
    Compute CLs value(s) for a given pyhf workspace.

    Example:

    .. code-block:: shell

        $ curl -sL https://git.io/JJYDE | pyhf cls

        \b
        {
            "CLs_exp": [
                0.07807427911686156,
                0.17472571775474618,
                0.35998495263681285,
                0.6343568235898907,
                0.8809947004472013
            ],
            "CLs_obs": 0.3599845631401915
        }
    """
    with click.open_file(workspace, 'r') as specstream:
        spec = json.load(specstream)

    ws = Workspace(spec)

    is_qtilde = teststat == 'qtilde'

    patches = [json.loads(click.open_file(pfile, 'r').read()) for pfile in patch]
    model = ws.model(
        measurement_name=measurement,
        patches=patches,
        modifier_settings={
            'normsys': {'interpcode': 'code4'},
            'histosys': {'interpcode': 'code4p'},
        },
    )

    # set the backend if not NumPy
    if backend in ['pytorch', 'torch']:
        set_backend("pytorch", precision="64b")
    elif backend in ['tensorflow', 'tf']:
        set_backend("tensorflow", precision="64b")
    elif backend in ['jax']:
        set_backend("jax")
    tensorlib, _ = get_backend()

    optconf = {k: v for item in optconf for k, v in item.items()}

    # set the new optimizer
    if optimizer:
        new_optimizer = getattr(optimize, optimizer) or getattr(
            optimize, f'{optimizer}_optimizer'
        )
        set_backend(tensorlib, new_optimizer(**optconf))

    result = hypotest(
        testpoi, ws.data(model), model, qtilde=is_qtilde, return_expected_set=True
    )
    result = {
        'CLs_obs': tensorlib.tolist(result[0]),
        'CLs_exp': [tensorlib.tolist(tensor) for tensor in result[-1]],
    }

    if output_file is None:
        click.echo(json.dumps(result, indent=4, sort_keys=True))
    else:
        with open(output_file, 'w+') as out_file:
            json.dump(result, out_file, indent=4, sort_keys=True)
        log.debug("Written to {0:s}".format(output_file))
