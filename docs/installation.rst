Installation
============

To install, we suggest first setting up a `virtual environment <https://virtualenvwrapper.readthedocs.io/en/latest/>`__

.. code-block:: console

    # Python3
    python3 -m venv pyhf


.. code-block:: console

    # Python2
    virtualenv --python=$(which python) pyhf

and activating it

.. code-block:: console

    source pyhf/bin/activate


Install from `PyPI <https://pypi.org/project/pyhf/>`__
------------------------------------------------------

with NumPy backend
++++++++++++++++++

.. code-block:: console

    pip install pyhf

with TensorFlow backend
+++++++++++++++++++++++

.. code-block:: console

    pip install pyhf[tensorflow]

with PyTorch backend
++++++++++++++++++++

.. code-block:: console

    pip install pyhf[torch]

with MXNet backend
++++++++++++++++++

.. code-block:: console

    pip install pyhf[mxnet]

with all backends
+++++++++++++++++

.. code-block:: console

    pip install pyhf[tensorflow,torch,mxnet]

Install from `GitHub <https://github.com/diana-hep/pyhf>`__
-----------------------------------------------------------

.. code-block:: console

    git clone https://github.com/diana-hep/pyhf.git
    cd pyhf

with NumPy backend
++++++++++++++++++

.. code-block:: console

    pip install --ignore-installed -U .

with TensorFlow backend
+++++++++++++++++++++++

.. code-block:: console

    pip install --ignore-installed -U .[tensorflow]

with PyTorch backend
++++++++++++++++++++

.. code-block:: console

    pip install --ignore-installed -U .[torch]

with MXNet backend
++++++++++++++++++

.. code-block:: console

    pip install --ignore-installed -U .[mxnet]

with all backends
+++++++++++++++++

.. code-block:: console

    pip install --ignore-installed -U .[tensorflow,torch,mxnet]
