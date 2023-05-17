==========
Developing
==========

Developer Environment
---------------------

To develop, we suggest using Python `virtual environments
<https://packaging.python.org/tutorials/installing-packages/#creating-virtual-environments>`__
together with ``pip`` and steered by `nox <https://github.com/wntrblm/nox>`__.
Once the virtual environment is activated and you have `SSH keys setup with GitHub
<https://docs.github.com/en/authentication/connecting-to-github-with-ssh>`__, clone the
repo from GitHub

.. code-block:: console

    git clone git@github.com:scikit-hep/pyhf

and install all necessary packages for development

.. code-block:: console

    python -m pip install --upgrade --editable '.[develop]'

Then setup the Git `pre-commit <https://pre-commit.com/>`__ hooks by running

.. code-block:: console

    pre-commit install

inside of the virtual environment.
`pre-commit.ci <https://pre-commit.ci/>`__ keeps the pre-commit hooks updated
through time, so pre-commit will automatically update itself when you run it
locally after the hooks were updated.

It is then suggested that you use ``nox`` to actually run all development operations
in "sessions" defined in ``noxfile.py``.
To list all of the available sessions run

.. code-block:: console

    nox --list

Linting
-------

Linting and code formatting is handled by ``pre-commit``.
To run the linting either run ``pre-commit``

.. code-block:: console

    pre-commit run --all-files

or use ``nox``

.. code-block:: console

    nox --session lint

Testing
-------

Writing tests
~~~~~~~~~~~~~

Data Files
^^^^^^^^^^

A function-scoped fixture called ``datadir`` exists for a given test module
which will automatically copy files from the associated test modules data
directory into a temporary directory for the given test execution. That is, for
example, if a test was defined in ``test_schema.py``, then data files located
in ``test_schema/`` will be copied to a temporary directory whose path is made
available by the ``datadir`` fixture. Therefore, one can do:

.. code-block:: python

    def test_patchset(datadir):
        data_file = open(datadir.join("test.txt"), encoding="utf-8")
        ...

which will load the copy of ``text.txt`` in the temporary directory. This also
works for parameterizations as this will effectively sandbox the file
modifications made.

Running with pytest
~~~~~~~~~~~~~~~~~~~

To run the test suite in full, from the top level of the repository run

.. code-block:: console

    pytest

More practically for most local testing you will not want to test the benchmarks,
contrib module, or notebooks, and so instead to test the core codebase a developer can run

.. code-block:: console

    nox --session tests --python 3.11

Contrib module matplotlib image tests
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To run the visualization tests for the ``contrib`` module with the ``pytest-mpl``
``pytest`` plugin run

.. code-block:: console

    nox --session tests --python 3.11 -- contrib

If the image files need to be regenerated, run the tests with the
``--mpl-generate-path=tests/contrib/baseline`` option or just run

.. code-block:: console

    nox --session regenerate

Doctest
^^^^^^^

``pyhf``'s configuration of ``pytest`` will automatically run ``doctest`` on all the
modules when the full test suite is run.
To run ``doctest`` on an individual module or file just run ``pytest`` on its path.
For example, to run ``doctest`` on the JAX backend run

.. code-block:: console

    pytest src/pyhf/tensor/jax_backend.py

Coverage
~~~~~~~~

To measure coverage for the codebase run the tests under ``coverage`` with

.. code-block:: console

    coverage run --module pytest

or pass ``coverage`` as a positional argument to the ``nox`` ``tests`` session

.. code-block:: console

    nox --session tests --python 3.11 -- coverage

Coverage Report
^^^^^^^^^^^^^^^

To generate a coverage report after running the tests under ``coverage`` run

.. code-block:: console

    coverage

or to also generate XML and HTML versions of the report run the coverage ``nox`` session

.. code-block:: console

    nox --session coverage

Documentation
-------------

To build the docs run

.. code-block:: console

    nox --session docs

To view the built docs locally, open the resulting ``docs/_build/html/index.html`` file
in a web browser or run

.. code-block:: console

    nox --session docs -- serve

Publishing
----------

Publishing to TestPyPI_ and PyPI_ is automated through the `PyPA's PyPI publish
GitHub Action <https://github.com/pypa/gh-action-pypi-publish>`__
and the ``pyhf`` `bump version GitHub Actions workflow
<https://github.com/scikit-hep/pyhf/blob/main/.github/workflows/bump-version.yml>`__.

Release Checklist
~~~~~~~~~~~~~~~~~

As part of the release process a checklist is required to be completed to make
sure steps aren't missed.
There is a GitHub Issue template for this that the maintainer in charge of the
release should step through and update if needed.

Release Tags
~~~~~~~~~~~~

A release tag can be created by a maintainer by using the `bump version GitHub Actions
workflow`_ through workflow dispatch.
The maintainer needs to:

* Select the semantic versioning (SemVer) type (major, minor, patch) of the release tag.
* Select if the release tag is a release candidate or not.
* Input the SemVer version number of the release tag.
* Select the branch to push the new release tag to.
* Select if to override the SemVer compatibility of the previous options (default
  is to run checks).
* Select if a dry run should be performed (default is to do a dry run to avoid accidental
  release tags).

The maintainer **should do a dry run first to make sure everything looks reasonable**.
Once they have done that, they can run the `bump version GitHub Actions workflow`_ which
will produce a new tag, bump the version of all files defined in `tbump.toml
<https://github.com/scikit-hep/pyhf/blob/main/tbump.toml>`__, and then commit and
push these changes and the tag back to the ``main`` branch.

Deployment
~~~~~~~~~~

The push of a tag to the repository will trigger a build of a sdist and wheel, and then
the deployment of them to TestPyPI_.

TestPyPI
^^^^^^^^

``pyhf`` tests packaging and distribution by publishing to TestPyPI_ in advance of
releases.
Installation of the latest test release from TestPyPI can be tested
by first installing ``pyhf`` normally, to ensure all dependencies are installed
from PyPI, and then upgrading ``pyhf`` to a test release from TestPyPI

.. code-block:: console

  python -m pip install pyhf
  python -m pip install --upgrade --extra-index-url https://test.pypi.org/simple/ --pre pyhf

.. note::

  This adds TestPyPI as `an additional package index to search
  <https://pip.pypa.io/en/stable/cli/pip_install/#cmdoption-extra-index-url>`__
  when installing.
  PyPI will still be the default package index ``pip`` will attempt to install
  from for all dependencies, but if a package has a release on TestPyPI that
  is a more recent release then the package will be installed from TestPyPI instead.
  Note that dev releases are considered pre-releases, so ``0.1.2`` is a "newer"
  release than ``0.1.2.dev3``.

PyPI
^^^^

Once the TestPyPI deployment has been examined, installed, and tested locally by the maintainers
final deployment to PyPI_ can be done by creating a GitHub Release:

#. From the ``pyhf`` `GitHub releases page <https://github.com/scikit-hep/pyhf/releases>`__
   select the `"Draft a new release" <https://github.com/scikit-hep/pyhf/releases/new>`__
   button.
#. Select the release tag that was just pushed, and set the release title to be the tag
   (e.g. ``v1.2.3``).
#. Use the "Auto-generate release notes" button to generate a skeleton of the release
   notes and then augment them with the preprepared release notes the release maintainer
   has written.
#. Select "This is a pre-release" if the release is a release candidate.
#. Select "Create a discussion for this release" if the release is a stable release.
#. Select "Publish release".

Once the release has been published to GitHub, the publishing workflow will build a
sdist and wheel, and then deploy them to PyPI_.

Context Files and Archive Metadata
----------------------------------

The ``.zenodo.json`` and ``codemeta.json`` files have the version number
automatically updated through ``tbump``, though their additional metadata
should be checked periodically by the dev team (probably every release).
The ``codemeta.json`` file can be generated automatically **from a PyPI install**
of ``pyhf`` using ``codemetapy``

.. code-block:: console

  codemetapy --no-extras pyhf > codemeta.json

though the ``author`` metadata will still need to be checked and revised by hand.
The ``.zenodo.json`` is currently generated by hand, so it is worth using
``codemeta.json`` as a guide to edit it.

.. _bump version GitHub Actions workflow: https://github.com/scikit-hep/pyhf/actions/workflows/bump-version.yml
.. _PyPI: https://pypi.org/project/pyhf/
.. _TestPyPI: https://test.pypi.org/project/pyhf/
