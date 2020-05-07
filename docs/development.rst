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

TestPyPI
~~~~~~~~

``pyhf`` tests packaging and distributing by publishing each commit to
``master`` to `TestPyPI <https://test.pypi.org/project/pyhf/>`__.
In addition, installation of the latest test release from TestPyPI can be tested
with

.. code-block:: bash

  python -m pip install --extra-index-url https://test.pypi.org/simple/ --pre pyhf

.. note::

  This adds TestPyPI as `an additional package index to search <https://pip.pypa.io/en/stable/reference/pip_install/#cmdoption-extra-index-url>`__
  when installing ``pyhf`` specifically.
  PyPI will still be the default package index ``pip`` will attempt to install
  from for all dependencies.

Publishing
----------

Publishing to `PyPI <https://pypi.org/project/pyhf/>`__ and `TestPyPI <https://test.pypi.org/project/pyhf/>`__
is automated through the `PyPA's PyPI publish GitHub Action <https://github.com/pypa/gh-action-pypi-publish>`__
and the ``pyhf`` `Tag Creator GitHub Actions workflow <https://github.com/scikit-hep/pyhf/blob/master/.github/workflows/tag.yml>`__.
A release can be created from any PR created by a core developer by adding a
``bumpversion`` tag to it that corresponds to the release type:
`major <https://github.com/scikit-hep/pyhf/labels/bumpversion%2Fmajor>`__,
`minor <https://github.com/scikit-hep/pyhf/labels/bumpversion%2Fminor>`__,
`patch <https://github.com/scikit-hep/pyhf/labels/bumpversion%2Fpatch>`__.
Once the PR is tagged with the label, the GitHub Actions bot will post a comment
with information on the actions it will take once the PR is merged. When the PR
has been reviewed, approved, and merged, the Tag Creator workflow will automatically
create a new release with ``bumpversion`` and then deploy the release to PyPI.
