# import sys
import os
import papermill as pm
import scrapbook as sb
import pytest


@pytest.fixture()
def common_kwargs(tmpdir):
    outputnb = tmpdir.join('output.ipynb')
    return {
        'output_path': str(outputnb),
        # 'kernel_name': f'python{sys.version_info.major}',
    }


def test_hello_world(common_kwargs):
    pm.execute_notebook('docs/examples/notebooks/hello-world.ipynb', **common_kwargs)


def test_xml_importexport(common_kwargs):
    pm.execute_notebook(
        'docs/examples/notebooks/XML_ImportExport.ipynb', **common_kwargs
    )


def test_statisticalanalysis(common_kwargs):
    # The Binder example uses specific relative paths
    cwd = os.getcwd()
    os.chdir(os.path.join(cwd, 'docs/examples/notebooks/binderexample'))
    pm.execute_notebook('StatisticalAnalysis.ipynb', **common_kwargs)
    os.chdir(cwd)


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
    cwd = os.getcwd()
    os.chdir(os.path.join(cwd, "docs/examples/notebooks"))
    pm.execute_notebook("pullplot.ipynb", **common_kwargs)
    os.chdir(cwd)


def test_impactplot(common_kwargs):
    # Change directories to make users not have to worry about paths to follow example
    cwd = os.getcwd()
    os.chdir(os.path.join(cwd, "docs/examples/notebooks"))
    pm.execute_notebook("ImpactPlot.ipynb", **common_kwargs)
    os.chdir(cwd)


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
