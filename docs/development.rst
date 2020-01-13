Developing
==========

To develop, we suggest using `virtual environments <https://packaging.python.org/tutorials/installing-packages/#creating-virtual-environments>`__ together with ``pip`` or using `pipenv <https://pipenv.readthedocs.io/en/latest/>`__. Once the environment is activated, clone the repo from GitHub

.. code-block:: console

    git clone https://github.com/scikit-hep/pyhf.git

and install all necessary packages for development

.. code-block:: console

    python -m pip install --ignore-installed -U -e .[complete]

Then setup the Git pre-commit hook for `Black <https://github.com/psf/black>`__  by running

.. code-block:: console

    pre-commit install

Publishing
----------

Publishing to `PyPI <https://pypi.org/project/pyhf/>`__ and `TestPyPI <https://test.pypi.org/project/pyhf/>`__
is automated through the `PyPA's PyPI publish GitHub Action <https://github.com/pypa/gh-action-pypi-publish>`__.
To publish a release to PyPI one simply needs to run

.. code-block:: console

    bumpversion [major|minor|patch]

to update the release version and get a tagged commit and then push the commit
and tag to :code:`master` with

.. code-block:: console

    git push origin master --tags
