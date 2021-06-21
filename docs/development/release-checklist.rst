Release Checklist
-----------------

When making a release PR for ``pyhf`` first copy the contents of this checklist
and paste it into a comment on the PR (with some small RST to Markdown
conversions).

Before Release
~~~~~~~~~~~~~~

* [ ] Verify that there is a release notes file for the release under ``docs/release-notes``.
* [ ] Verify that the release notes files correctly summarizes all development
  changes since the last release.
* [ ] Draft email to ``pyhf-announcements`` mailing list that summarizes the
  main points of the release notes.
* [ ] Update the checklist file in the ``docs`` directory if there are revisions.


Once Release PR is Merged
~~~~~~~~~~~~~~~~~~~~~~~~~

* [ ] Create a GitHub release from the generated PR tag

After Release
~~~~~~~~~~~~~

Update things post release

* [ ] Verify that the release is installable from both PyPI and Conda-forge.
* [ ] Tweet the release out on both personal and team Twitter accounts.
* [ ] Update the tutorial.
