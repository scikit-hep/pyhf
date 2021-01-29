"""The inference CLI group."""
import logging

import click
import json

from ..utils import EqDelimStringParamType
from ..infer import hypotest
from ..infer import mle
from ..workspace import Workspace
from .. import get_backend, set_backend, optimize

log = logging.getLogger(__name__)


@click.group(name='infer')
def cli():
    """Infererence CLI group."""


@cli.command()
@click.argument("workspace", default="-")
@click.option(
    "--output-file",
    help="The location of the output json file. If not specified, prints to screen.",
    default=None,
)
@click.option("--measurement", default=None)
@click.option("-p", "--patch", multiple=True)
@click.option(
    "--value",
    help="Flag for returning the fitted value of the objective function.",
    default=False,
    is_flag=True,
)
@click.option(
    "--backend",
    type=click.Choice(["numpy", "pytorch", "tensorflow", "jax", "np", "torch", "tf"]),
    help="The tensor backend used for the calculation.",
    default="numpy",
)
@click.option(
    "--optimizer",
    type=click.Choice(["scipy", "minuit"]),
    help="The optimizer used for the calculation.",
    default="scipy",
)
@click.option("--optconf", type=EqDelimStringParamType(), multiple=True)
def fit(
    workspace,
    output_file,
    measurement,
    patch,
    value,
    backend,
    optimizer,
    optconf,
):
    """
    Perform a maximum likelihood fit for a given pyhf workspace.

    Example:

    .. code-block:: shell

        $ curl -sL https://git.io/JJYDE | pyhf fit --value

        \b
        {
            "mle_parameters": {
                "mu": [
                    0.00012198211767276228
                ],
                "uncorr_bkguncrt": [
                    0.9999644178569168,
                    0.999895631468898
                ]
            },
            "nll": 11.598183353771212
        }
    """
    # set the backend if not NumPy
    if backend in ["pytorch", "torch"]:
        set_backend("pytorch", precision="64b")
    elif backend in ["tensorflow", "tf"]:
        set_backend("tensorflow", precision="64b")
    elif backend in ["jax"]:
        set_backend("jax")
    tensorlib, _ = get_backend()

    optconf = {k: v for item in optconf for k, v in item.items()}

    # set the new optimizer
    if optimizer:
        new_optimizer = getattr(optimize, optimizer) or getattr(
            optimize, f"{optimizer}_optimizer"
        )
        set_backend(tensorlib, new_optimizer(**optconf))

    with click.open_file(workspace, "r") as specstream:
        spec = json.load(specstream)
    ws = Workspace(spec)
    patches = [json.loads(click.open_file(pfile, "r").read()) for pfile in patch]

    model = ws.model(
        measurement_name=measurement,
        patches=patches,
    )
    data = ws.data(model)

    fit_result = mle.fit(data, model, return_fitted_val=value)

    _pars = fit_result if not value else fit_result[0]
    bestfit_pars = {
        k: tensorlib.tolist(_pars[v["slice"]]) for k, v in model.config.par_map.items()
    }

    result = {"mle_parameters": bestfit_pars}
    if value:
        result["nll"] = tensorlib.tolist(fit_result[-1])

    if output_file is None:
        click.echo(json.dumps(result, indent=4, sort_keys=True))
    else:
        with open(output_file, "w+") as out_file:
            json.dump(result, out_file, indent=4, sort_keys=True)
        log.debug(f"Written to {output_file:s}")


@cli.command()
@click.argument('workspace', default='-')
@click.option(
    '--output-file',
    help='The location of the output json file. If not specified, prints to screen.',
    default=None,
)
@click.option('--measurement', default=None)
@click.option('-p', '--patch', multiple=True)
@click.option('--test-poi', default=1.0)
@click.option('--test-stat', type=click.Choice(['q', 'qtilde']), default='qtilde')
@click.option(
    '--calctype', type=click.Choice(['asymptotics', 'toybased']), default='asymptotics'
)
@click.option(
    '--backend',
    type=click.Choice(['numpy', 'pytorch', 'tensorflow', 'jax', 'np', 'torch', 'tf']),
    help='The tensor backend used for the calculation.',
    default='numpy',
)
@click.option(
    "--optimizer",
    type=click.Choice(["scipy", "minuit"]),
    help="The optimizer used for the calculation.",
    default="scipy",
)
@click.option('--optconf', type=EqDelimStringParamType(), multiple=True)
def cls(
    workspace,
    output_file,
    measurement,
    patch,
    test_poi,
    test_stat,
    backend,
    optimizer,
    calctype,
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
                0.07807398485749008,
                0.17472524541749054,
                0.3599843354244575,
                0.634356267897945,
                0.8809944338156985
            ],
            "CLs_obs": 0.3599843354244575
        }
    """
    with click.open_file(workspace, 'r') as specstream:
        spec = json.load(specstream)

    ws = Workspace(spec)

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
        test_poi,
        ws.data(model),
        model,
        test_stat=test_stat,
        calctype=calctype,
        return_expected_set=True,
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
        log.debug(f"Written to {output_file:s}")
