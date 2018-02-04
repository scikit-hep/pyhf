import os
import papermill as pm

def test_notebooks():
    pm.execute_notebook('examples/notebooks/ShapeFactor.ipynb','/dev/null', kernel_name='python')
