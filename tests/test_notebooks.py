import sys
from pathlib import Path

import papermill as pm
import pytest
import scrapbook as sb


@pytest.fixture()
def common_kwargs(tmpdir):
    outputnb = tmpdir.join('output.ipynb')
    return {
        'output_path': Path(outputnb),
        'kernel_name': f'python{sys.version_info.major}',
        'progress_bar': False,
    }


def test_hello_world(common_kwargs):
    pm.execute_notebook('docs/examples/notebooks/hello-world.ipynb', **common_kwargs)


def test_xml_importexport(common_kwargs):
    pm.execute_notebook(
        'docs/examples/notebooks/XML_ImportExport.ipynb', **common_kwargs
    )


def test_statisticalanalysis(common_kwargs):
    # The Binder example uses specific relative paths
    execution_dir = Path.cwd() / "docs" / "examples" / "notebooks" / "binderexample"
    pm.execute_notebook(
        execution_dir / "StatisticalAnalysis.ipynb", cwd=execution_dir, **common_kwargs
    )


def test_shapefactor(common_kwargs):
    pm.execute_notebook('docs/examples/notebooks/ShapeFactor.ipynb', **common_kwargs)


def test_multichannel_coupled_histos(common_kwargs):
    pm.execute_notebook(
        'docs/examples/notebooks/multichannel-coupled-histo.ipynb',
        parameters={'validation_datadir': 'validation/data'},
        **common_kwargs,
    )


def test_multibinpois(common_kwargs):
    pm.execute_notebook(
        'docs/examples/notebooks/multiBinPois.ipynb',
        parameters={'validation_datadir': 'validation/data'},
        **common_kwargs,
    )
    nb = sb.read_notebook(common_kwargs['output_path'])
    assert nb.scraps['number_2d_successpoints'].data > 200


def test_pullplot(common_kwargs):
    # Change directories to make users not have to worry about paths to follow example
    execution_dir = Path.cwd() / "docs" / "examples" / "notebooks"
    pm.execute_notebook(
        execution_dir / "pullplot.ipynb", cwd=execution_dir, **common_kwargs
    )


def test_impactplot(common_kwargs):
    # Change directories to make users not have to worry about paths to follow example
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
