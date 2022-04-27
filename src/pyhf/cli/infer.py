"""The inference CLI group."""
import logging

import click
import json

from pyhf.utils import EqDelimStringParamType
from pyhf.infer import hypotest
from pyhf.infer import mle
from pyhf.workspace import Workspace
from pyhf import get_backend, set_backend, optimize

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

        $ curl -sL https://raw.githubusercontent.com/scikit-hep/pyhf/master/docs/examples/json/2-bin_1-channel.json | pyhf fit --value

        \b
        {
            "mle_parameters": {
                "mu": [
                    0.00017298628839781602
                ],
                "uncorr_bkguncrt": [
                    1.0000015671710816,
                    0.9999665895859197
                ]
            },
            "twice_nll": 23.19636590468879
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

    optconf = {
        opt_name: opt_value for item in optconf for opt_name, opt_value in item.items()
    }

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
        paramset_name: tensorlib.tolist(_pars[paramset_spec["slice"]])
        for paramset_name, paramset_spec in model.config.par_map.items()
    }

    result = {"mle_parameters": bestfit_pars}
    if value:
        result["twice_nll"] = tensorlib.tolist(fit_result[-1])

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

        $ curl -sL https://raw.githubusercontent.com/scikit-hep/pyhf/master/docs/examples/json/2-bin_1-channel.json | pyhf cls

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

    optconf = {
        opt_name: opt_value for item in optconf for opt_name, opt_value in item.items()
    }

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
