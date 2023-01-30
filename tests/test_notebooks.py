import os
import sys
from pathlib import Path

import nbformat
import papermill as pm
import pytest
import scrapbook as sb
from nbclient import NotebookClient

# Avoid hanging on with ipywidgets interact by using non-gui backend
os.environ["MPLBACKEND"] = "agg"


@pytest.fixture()
def common_kwargs(tmpdir):
    outputnb = tmpdir.join('output.ipynb')
    return {
        'output_path': str(outputnb),
        'kernel_name': f'python{sys.version_info.major}',
        'progress_bar': False,
    }


# def test_hello_world(common_kwargs):
#     pm.execute_notebook('docs/examples/notebooks/hello-world.ipynb', **common_kwargs)


def get_notebook_client(notebook, commons, execution_dir=None):
    if execution_dir:
        commons["resources"]["metadata"]["path"] = execution_dir
    else:
        execution_dir = commons["resources"]["metadata"]["path"]

    test_notebook = nbformat.read(execution_dir / notebook, as_version=4)
    return NotebookClient(test_notebook, **commons)


# c.f. https://nbclient.readthedocs.io/en/latest/client.html
@pytest.fixture()
def commons():
    execution_dir = Path.cwd() / "docs" / "examples" / "notebooks"
    return {
        "timeout": 600,
        "kernel_name": "python3",
        "reset_kc": True,
        "resources": {"metadata": {"path": execution_dir}},
    }


def test_hello_world(commons):
    client = get_notebook_client("hello-world.ipynb", commons)
    assert client.execute()


def test_xml_importexport(commons):
    client = get_notebook_client("XML_ImportExport.ipynb", commons)
    assert client.execute()


def test_statisticalanalysis(commons):
    # The Binder example uses specific relative paths
    client = get_notebook_client(
        "StatisticalAnalysis.ipynb",
        commons,
        execution_dir=commons["resources"]["metadata"]["path"] / "binderexample",
    )
    assert client.execute()


def test_shapefactor(commons):
    client = get_notebook_client("ShapeFactor.ipynb", commons)
    assert client.execute()


def test_multichannel_coupled_histos(common_kwargs):
    pm.execute_notebook(
        'docs/examples/notebooks/multichannel-coupled-histo.ipynb',
        parameters={"validation_datadir": str(Path.cwd() / "validation" / "data")},
        **common_kwargs,
    )


def test_multibinpois(common_kwargs):
    pm.execute_notebook(
        'docs/examples/notebooks/multiBinPois.ipynb',
        parameters={"validation_datadir": str(Path.cwd() / "validation" / "data")},
        **common_kwargs,
    )
    nb = sb.read_notebook(common_kwargs['output_path'])
    assert nb.scraps['number_2d_successpoints'].data > 200


def test_pullplot(common_kwargs):
    execution_dir = Path.cwd() / "docs" / "examples" / "notebooks"
    pm.execute_notebook(
        execution_dir / "pullplot.ipynb", cwd=execution_dir, **common_kwargs
    )


def test_impactplot(common_kwargs):
    execution_dir = Path.cwd() / "docs" / "examples" / "notebooks"
    pm.execute_notebook(
        execution_dir / "ImpactPlot.ipynb", cwd=execution_dir, **common_kwargs
    )


def test_toys(common_kwargs):
    pm.execute_notebook('docs/examples/notebooks/toys.ipynb', **common_kwargs)


def test_learn_interpolationcodes(common_kwargs):
    pm.execute_notebook(
        'docs/examples/notebooks/learn/InterpolationCodes.ipynb', **common_kwargs
    )


def test_learn_tensorizinginterpolations(common_kwargs):
    pm.execute_notebook(
        'docs/examples/notebooks/learn/TensorizingInterpolations.ipynb', **common_kwargs
    )


def test_learn_using_calculators(common_kwargs):
    pm.execute_notebook(
        "docs/examples/notebooks/learn/UsingCalculators.ipynb", **common_kwargs
    )
