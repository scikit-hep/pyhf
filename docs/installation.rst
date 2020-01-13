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

    python -m pip install pyhf[tensorflow]

... with PyTorch backend
++++++++++++++++++++++++

.. code-block:: console

    python -m pip install pyhf[torch]

... with JAX backend
++++++++++++++++++++

.. code-block:: console

    python -m pip install pyhf[jax]

... with all backends
+++++++++++++++++++++

.. code-block:: console

    python -m pip install pyhf[backends]


... with xml import/export functionality
++++++++++++++++++++++++++++++++++++++++

.. code-block:: console

    python -m pip install pyhf[xmlio]


Install latest development version from `GitHub <https://github.com/scikit-hep/pyhf>`__...
------------------------------------------------------------------------------------------

... with NumPy backend
++++++++++++++++++++++

.. code-block:: console

    python -m pip install --ignore-installed -U "git+https://github.com/scikit-hep/pyhf.git#egg=pyhf"

... with TensorFlow backend
+++++++++++++++++++++++++++

.. code-block:: console

    python -m pip install --ignore-installed -U "git+https://github.com/scikit-hep/pyhf.git#egg=pyhf[tensorflow]"

... with PyTorch backend
++++++++++++++++++++++++

.. code-block:: console

    python -m pip install --ignore-installed -U "git+https://github.com/scikit-hep/pyhf.git#egg=pyhf[torch]"

... with JAX backend
++++++++++++++++++++++

.. code-block:: console

    python -m pip install --ignore-installed -U "git+https://github.com/scikit-hep/pyhf.git#egg=pyhf[jax]"

... with all backends
+++++++++++++++++++++

.. code-block:: console

    python -m pip install --ignore-installed -U "git+https://github.com/scikit-hep/pyhf.git#egg=pyhf[backends]"


... with xml import/export functionality
++++++++++++++++++++++++++++++++++++++++

.. code-block:: console

    python -m pip install --ignore-installed -U "git+https://github.com/scikit-hep/pyhf.git#egg=pyhf[xmlio]"


Updating :code:`pyhf`
---------------------

Rerun the installation command. As the upgrade flag, :code:`-U`, is used then the libraries will be updated.
