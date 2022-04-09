==========
Developing
==========

Developer Environment
---------------------

To develop, we suggest using `virtual environments <https://packaging.python.org/tutorials/installing-packages/#creating-virtual-environments>`__ together with ``pip`` or using `pipenv <https://pipenv.readthedocs.io/en/latest/>`__. Once the environment is activated, clone the repo from GitHub

.. code-block:: console

    git clone https://github.com/scikit-hep/pyhf.git

and install all necessary packages for development

.. code-block:: console

    python -m pip install --upgrade --editable .[complete]

Then setup the Git pre-commit hook for `Black <https://github.com/psf/black>`__  by running

.. code-block:: console

    pre-commit install

as the ``rev`` gets updated through time to track changes of different hooks,
simply run

.. code-block:: console

    pre-commit autoupdate

to have pre-commit install the new version.

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
        data_file = open(datadir.join("test.txt"))
        ...

which will load the copy of ``text.txt`` in the temporary directory. This also
works for parameterizations as this will effectively sandbox the file
modifications made.

TestPyPI
~~~~~~~~

``pyhf`` tests packaging and distributing by publishing in advance of releases
to TestPyPI_.
Installation of the latest test release from TestPyPI can be tested
by first installing ``pyhf`` normally, to ensure all dependencies are installed
from PyPI, and then upgrading ``pyhf`` to a test release from TestPyPI

.. code-block:: bash

  python -m pip install pyhf
  python -m pip install --upgrade --extra-index-url https://test.pypi.org/simple/ --pre pyhf

.. note::

  This adds TestPyPI as `an additional package index to search <https://pip.pypa.io/en/stable/reference/pip_install/#cmdoption-extra-index-url>`__
  when installing.
  PyPI will still be the default package index ``pip`` will attempt to install
  from for all dependencies, but if a package has a release on TestPyPI that
  is a more recent release then the package will be installed from TestPyPI instead.
  Note that dev releases are considered pre-releases, so ``0.1.2`` is a "newer"
  release than ``0.1.2.dev3``.

Publishing
----------

Publishing to PyPI_ and TestPyPI_ is automated through the `PyPA's PyPI publish
GitHub Action <https://github.com/pypa/gh-action-pypi-publish>`__
and the ``pyhf`` `Bump version GitHub Actions workflow
<https://github.com/scikit-hep/pyhf/blob/master/.github/workflows/bump-version.yml>`__.


Release Checklist
~~~~~~~~~~~~~~~~~

As part of the release process a checklist is required to be completed to make
sure steps aren't missed.
There is a GitHub Issue template for this that the maintainer in charge of the
release should step through and update if needed.

Release Tags
~~~~~~~~~~~~

A release tag can be created by a maintainer by using the bump version workflow
through GitHub Actions workflow dispatch.
The maintainer needs to:

* Select the semantic versioning (SemVer) type (major, minor, patch) of the release tag.
* Select if the release tag is a release candidate or not.
* Input the SemVer version number of the release tag.
* Select if to override the SemVer compatibility of the previous options (default
  is to run checks).
* Select if a dry run should be performed (default is to do a dry run to avoid accidental
  release tags).

The maintainer should do a dry run first to make sure everything looks reasonable.
Once they have done that, they can run the bump version workflow which will produce
a new tag, bump the version of all files defined in `tbump.toml
<https://github.com/scikit-hep/pyhf/blob/master/tbump.toml>`__, and then commit and
push these changes and the tag back to the ``master`` branch.

Deployment
~~~~~~~~~~

The push of a tag to the repository will trigger a build of a sdist and wheel, and then
the deployment of them to TestPyPI_.
Once the deployment has been examined, installed, and tested locally by the maintainers
final deployment to PyPI_ can be done.

Releases are performed through GitHub Releases.

* From the ``pyhf`` `GitHub releases page <https://github.com/scikit-hep/pyhf/releases>`__
  select the `"Draft a new release" <https://github.com/scikit-hep/pyhf/releases/new>`__
  button.
* Select the release tag that was just pushed, and set the release title to be the tag
  (e.g. `v1.2.3`).
* Use the "Auto-generate release notes" button to generate a skeleton of the release
  notes and then augment them with the preprepared release notes the release maintainer
  has written.
* Select "This is a pre-release" if the release is a release candidate.
* Select "Create a discussion for this release" if the release is a stable release.
* Select "Publish release".

Once the release has been published to GitHub, the publishing workflow will build a
sdist and wheel, and then deploy them to PyPI_.

Context Files and Archive Metadata
----------------------------------

The ``.zenodo.json`` and ``codemeta.json`` files have the version number
automatically updated through ``bump2version``, though their additional metadata
should be checked periodically by the dev team (probably every release).
The ``codemeta.json`` file can be generated automatically **from a PyPI install**
of ``pyhf`` using ``codemetapy``

.. code-block:: bash

  codemetapy --no-extras pyhf > codemeta.json

though the ``author`` metadata will still need to be checked and revised by hand.
The ``.zenodo.json`` is currently generated by hand, so it is worth using
``codemeta.json`` as a guide to edit it.

.. _PyPI: https://pypi.org/project/pyhf/
.. _TestPyPI: https://test.pypi.org/project/pyhf/
