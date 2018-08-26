import sys
import papermill as pm


def test_notebooks(tmpdir):
    outputnb = tmpdir.join('output.ipynb')
    common_kwargs = {
        'output': str(outputnb),
        'kernel_name': 'python{}'.format(sys.version_info.major)
    }

    pm.execute_notebook(
        'docs/examples/notebooks/hello-world.ipynb', **common_kwargs)

    pm.execute_notebook(
        'docs/examples/notebooks/ShapeFactor.ipynb', **common_kwargs)
    pm.execute_notebook('docs/examples/notebooks/multichannel-coupled-histo.ipynb',
                        parameters={'validation_datadir': 'validation/data'},
                        **common_kwargs)
    pm.execute_notebook('docs/examples/notebooks/multiBinPois.ipynb',
                        parameters={'validation_datadir': 'validation/data'},
                        **common_kwargs)

    nb = pm.read_notebook(str(outputnb))
    assert nb.data['number_2d_successpoints'] > 200
