import os
import papermill as pm
import sys

def test_notebooks():
    common_kwargs = {'output': '/dev/null', 'kernel_name':'python{}'.format(sys.version_info.major)}

    pm.execute_notebook('examples/notebooks/ShapeFactor.ipynb',**common_kwargs)
    pm.execute_notebook('examples/notebooks/multichannel-coupled-histo.ipynb',parameters = {
        'validation_datadir': 'validation/data'
    }, **common_kwargs)
    pm.execute_notebook('examples/notebooks/multiBinPois.ipynb',parameters = {
        'validation_datadir': 'validation/data'
    }, **common_kwargs)
