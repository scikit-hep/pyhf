Release Checklist
-----------------

When making a release PR for ``pyhf`` first **copy the contents of this checklist
and paste it into a comment on the PR** (with some small RST to Markdown
conversions).

Before Release
~~~~~~~~~~~~~~

* [ ] Migrate any unresolved Issues or PRs from the
  `release GitHub project board <https://github.com/scikit-hep/pyhf/projects>`_
  to a new project board.
* [ ] Verify that there is a release notes file for the release under ``docs/release-notes``.
* [ ] Verify that the release notes files correctly summarize all development
  changes since the last release.
* [ ] Draft email to ``pyhf-announcements`` mailing list that summarizes the
  main points of the release notes and circulate it for development team
  approval.
* [ ] Update the checklist file in the ``docs`` directory if there are revisions.
* [ ] Verify that the project README is displaying correctly on
  `TestPyPI <https://test.pypi.org/project/pyhf/>`_.
* [ ] Add any new use citations to the Use and Citations page.
* [ ] Update the `pypa/gh-action-pypi-publish <https://github.com/pypa/gh-action-pypi-publish>`_
  GitHub Action used for deployment to TestPyPI and PyPI to the latest stable
  release.


Once Release PR is Merged
~~~~~~~~~~~~~~~~~~~~~~~~~

* [ ] Watch the CI to ensure that the deployment to PyPI is successful.
* [ ] Create a `GitHub release <https://github.com/scikit-hep/pyhf/releases>`_
  from the generated PR tag and copy the release notes published to the GitHub
  release page.
  The creation of the GitHub release triggers all other release related activities.
* [ ] Verify there is a new `Zenodo DOI <https://doi.org/10.5281/zenodo.1169739>`_
  minted for the release.
* [ ] Verify that a Binder has properly built for the new release.
* [ ] Watch for a GitHub notification that there is an automatic PR to the
  `Conda-forge feedstock <https://github.com/conda-forge/pyhf-feedstock>`_.
  This may take multiple hours to happen.
  If there are any changes needed to the Conda-forge release make them **from a
  personal account** and not from an organization account to have workflows
  properly trigger.

After Release
~~~~~~~~~~~~~

* [ ] Verify that the release is installable from both PyPI and Conda-forge.
* [ ] Send the drafted ``pyhf-announcements`` email out from the
  ``pyhf-announcements`` account email.
* [ ] Tweet the release out on both personal and team Twitter accounts.
* [ ] Announce the release on the `Scikit-HEP community
  Gitter <https://gitter.im/Scikit-HEP/community>`_.
* [ ] Make a release for the ``pyhf`` `tutorial <https://github.com/pyhf/pyhf-tutorial>`_
  corresponding to the **previous release** number.
  This release represents the last version of the tutorial that is guaranteed
  to work with previous release API.
* [ ] Update the tutorial to use the new release number and API.
* [ ] Make a PR to use the new release in the `CUDA enabled Docker
  images <https://github.com/pyhf/cuda-images>`_.
