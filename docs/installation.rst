..  _installation:

Installation
============

To install, we suggest first setting up a `virtual environment <https://packaging.python.org/tutorials/installing-packages/#creating-virtual-environments>`__

.. code-block:: console

    # Python3
    python3 -m venv pyhf

and activating it

.. code-block:: console

    source pyhf/bin/activate


Install latest stable release from `PyPI <https://pypi.org/project/pyhf/>`__...
-------------------------------------------------------------------------------

... with NumPy backend
++++++++++++++++++++++

.. code-block:: console

    python -m pip install pyhf

... with TensorFlow backend
+++++++++++++++++++++++++++

.. code-block:: console

    python -m pip install 'pyhf[tensorflow]'

... with PyTorch backend
++++++++++++++++++++++++

.. code-block:: console

    python -m pip install 'pyhf[torch]'

... with JAX backend
++++++++++++++++++++

.. code-block:: console

    python -m pip install 'pyhf[jax]'

... with all backends
+++++++++++++++++++++

.. code-block:: console

    python -m pip install 'pyhf[backends]'


... with xml import/export functionality
++++++++++++++++++++++++++++++++++++++++

.. code-block:: console

    python -m pip install 'pyhf[xmlio]'


Install latest development version from `GitHub <https://github.com/scikit-hep/pyhf>`__...
------------------------------------------------------------------------------------------

... with NumPy backend
++++++++++++++++++++++

.. code-block:: console

    python -m pip install --upgrade 'pyhf@git+https://github.com/scikit-hep/pyhf.git'

... with TensorFlow backend
+++++++++++++++++++++++++++

.. code-block:: console

    python -m pip install --upgrade 'pyhf[tensorflow]@git+https://github.com/scikit-hep/pyhf.git'

... with PyTorch backend
++++++++++++++++++++++++

.. code-block:: console

    python -m pip install --upgrade 'pyhf[torch]@git+https://github.com/scikit-hep/pyhf.git'

... with JAX backend
++++++++++++++++++++++

.. code-block:: console

    python -m pip install --upgrade 'pyhf[jax]@git+https://github.com/scikit-hep/pyhf.git'

... with all backends
+++++++++++++++++++++

.. code-block:: console

    python -m pip install --upgrade 'pyhf[backends]@git+https://github.com/scikit-hep/pyhf.git'


... with xml import/export functionality
++++++++++++++++++++++++++++++++++++++++

.. code-block:: console

    python -m pip install --upgrade 'pyhf[xmlio]@git+https://github.com/scikit-hep/pyhf.git'


Updating :code:`pyhf`
---------------------

Rerun the installation command. As the upgrade flag (:code:`-U`, :code:`--upgrade`) is used then the libraries will be updated.
