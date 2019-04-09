import sys
import os
import papermill as pm


def test_notebooks(tmpdir):
    outputnb = tmpdir.join('output.ipynb')
    common_kwargs = {
        'output_path': str(outputnb),
        'kernel_name': 'python{}'.format(sys.version_info.major),
    }

    pm.execute_notebook('docs/examples/notebooks/hello-world.ipynb', **common_kwargs)

    pm.execute_notebook(
        'docs/examples/notebooks/XML_ImportExport.ipynb', **common_kwargs
    )

    if sys.version_info.major > 2:
        # The Binder example uses specific relative paths
        cwd = os.getcwd()
        os.chdir(os.path.join(cwd, 'docs/examples/notebooks/binderexample'))
        pm.execute_notebook('StatisticalAnalysis.ipynb', **common_kwargs)
        os.chdir(cwd)

    pm.execute_notebook(
        'docs/examples/notebooks/learn/InterpolationCodes.ipynb', **common_kwargs
    )

    pm.execute_notebook('docs/examples/notebooks/ShapeFactor.ipynb', **common_kwargs)
    pm.execute_notebook(
        'docs/examples/notebooks/multichannel-coupled-histo.ipynb',
        parameters={'validation_datadir': 'validation/data'},
        **common_kwargs
    )
    pm.execute_notebook(
        'docs/examples/notebooks/multiBinPois.ipynb',
        parameters={'validation_datadir': 'validation/data'},
        **common_kwargs
    )

    nb = pm.read_notebook(str(outputnb))
    assert nb.data['number_2d_successpoints'] > 200
