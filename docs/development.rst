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

Testing
-------

Data Files
~~~~~~~~~~

A function-scoped fixture called ``datadir`` exists for a given test module
which will automatically copy files from the associated test modules data
directory into a temporary directory for the given test execution. That is, for
example, if a test was defined in ``test_schema.py``, then data files located
in ``test_schema/`` will be copied to a temporary directory whose path is made
available by the ``datadir`` fixture. Therefore, one can do:

.. code-block:: python

    def test_patchset(datadir):
        data_file = open(datadir.join('test.txt'))
        ...

which will load the copy of ``text.txt`` in the temporary directory. This also
works for parameterizations as this will effectively sandbox the file
modifications made.
