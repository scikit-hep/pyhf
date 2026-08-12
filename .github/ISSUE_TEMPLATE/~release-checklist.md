---
name: ✅  Release Checklist (Maintainers Only)
about: Checklist for core developers to complete as part of making a release

---
# Release Checklist

## Before Release

* [ ] Update the checklist Issue template in the [``.github/ISSUE_TEMPLATE``](https://github.com/scikit-hep/pyhf/tree/main/.github/ISSUE_TEMPLATE) directory if there are revisions.
* [ ] Migrate any unresolved Issues or PRs from the [release GitHub project board](https://github.com/scikit-hep/pyhf/projects/) to a new project board.
* [ ] Verify that there is a release notes file for the release under [``docs/release-notes``](https://github.com/scikit-hep/pyhf/tree/main/docs/release-notes).
* [ ] Verify that the release notes files correctly summarize all development changes since the last release.
* [ ] Add any new use citations or published statistical models to the [Use and Citations page][citations_page].
* [ ] Verify that the citations on the [Use and Citations page][citations_page] are up to date with their current [INSPIRE](https://inspirehep.net/) record. Checking the [Dimensions listing of publication citations](https://app.dimensions.ai/discover/publication?or_subset_publication_citations=pub.1135154020) can be helpful to catch citations that are now journal publications.
* [ ] Update the [pypa/gh-action-pypi-publish](https://github.com/pypa/gh-action-pypi-publish) GitHub Action used for deployment to TestPyPI and PyPI to the latest stable release.
* [ ] Make a release to [TestPyPI][TestPyPI_pyhf] using the [workflow dispatch event trigger](https://github.com/scikit-hep/pyhf/actions/workflows/publish-package.yml).
* [ ] Verify that the project README is displaying correctly on [TestPyPI][TestPyPI_pyhf].
* [ ] Draft email to [``pyhf-announcements`` mailing list](https://groups.google.com/group/pyhf-announcements/) that summarizes the main points of the release notes and circulate it for development team approval.

[TestPyPI_pyhf]: https://test.pypi.org/project/pyhf/
[citations_page]: https://scikit-hep.org/pyhf/citations.html

## Create Release Tag

* [ ] Run the [Prepare release](https://github.com/scikit-hep/pyhf/actions/workflows/release-prepare.yml) GitHub Actions workflow on the release branch, entering the version of the new release.
* [ ] Review the release preparation pull request the workflow opens: verify the new version and the diff of the bumped files, and wait for CI to pass.
* [ ] Merge the release preparation pull request.
* [ ] Run the [Tag release](https://github.com/scikit-hep/pyhf/actions/workflows/release-tag.yml) GitHub Actions workflow on the release branch and approve the `release-tag` environment deployment.
* [ ] Verify the release tag was pushed to the correct branch.
* [ ] Watch the CI to verify all tag based jobs finish correctly.
* [ ] Verify the release for the tag on [TestPyPI][TestPyPI_pyhf] looks correct.

## After Release Tag Pushed To GitHub

* [ ] Create a [GitHub release](https://github.com/scikit-hep/pyhf/releases) from the new release tag and copy the release notes published to the GitHub release page. The creation of the GitHub release triggers all other release related activities.
   - [ ] Create a corresponding [announcement GitHub Discussion](https://github.com/scikit-hep/pyhf/discussions/categories/announcements) for the release.
* [ ] Watch the CI to ensure that the deployment to [PyPI](https://pypi.org/project/pyhf/) is successful.
* [ ] Verify Docker images with the correct tags have been deployed to all container image registries.
   - [ ] [Docker Hub](https://hub.docker.com/r/pyhf/pyhf/tags)
   - [ ] [OSG Harbor](https://hub.opensciencegrid.org/harbor/projects/866/repositories/pyhf/)
   - [ ] [CERN Harbor](https://registry.cern.ch/harbor/projects/3550/repositories/pyhf/artifacts-tab)
* [ ] Verify there is a new [Zenodo DOI](https://doi.org/10.5281/zenodo.1169739) minted for the release.
   - [ ] Verify that the new release archive metadata on Zenodo matches is being picked up as expected from [`CITATION.cff`](https://github.com/scikit-hep/pyhf/blob/main/CITATION.cff).
* [ ] Verify that a Binder has properly built for the new release.
* [ ] Watch for a GitHub notification that there is an automatic PR to the [conda-forge feedstock](https://github.com/conda-forge/pyhf-feedstock). This may take multiple hours to happen. If there are any changes needed to the conda-forge release make them **from a personal account** and not from an organization account to have workflows properly trigger.
   - [ ] Verify the requirements in the [conda-forge feedstock](https://github.com/conda-forge/pyhf-feedstock) recipe `meta.yaml` match those in `pyproject.toml`.

## After Release

* [ ] Verify that the release is installable from both [PyPI](https://pypi.org/project/pyhf/) and [conda-forge](https://github.com/conda-forge/pyhf-feedstock).
* [ ] Send the drafted [``pyhf-announcements``](https://groups.google.com/group/pyhf-announcements/) email out from the ``pyhf-announcements`` account email.
* [ ] Share the release on both personal and team social media accounts.
* [ ] Forward port the release notes and the `tbump.toml` version information from the release branch to the default branch.
   - c.f. PR https://github.com/scikit-hep/pyhf/pull/2217 and PR https://github.com/scikit-hep/pyhf/pull/2218 as examples from `pyhf` `v0.7.2`.
* [ ] Announce the release on the [Scikit-HEP community Gitter](https://gitter.im/Scikit-HEP/community).
* [ ] Make a release for the [`pyhf` tutorial](https://github.com/pyhf/pyhf-tutorial/releases) corresponding to the **previous release** number. This release represents the last version of the tutorial that is guaranteed to work with previous release API.
* [ ] Update the [tutorial](https://github.com/pyhf/pyhf-tutorial) to use the new release number and API.
* [ ] Make a PR to use the new release in the [CUDA enabled Docker images](https://github.com/pyhf/cuda-images).
* [ ] Open a ticket on the CERN [Software Process and Infrastructure JIRA](https://sft.its.cern.ch/jira/browse/SPI) to update the version of `pyhf` available in the next LCG release.
   - c.f. the [`v0.7.3` request ticket](https://sft.its.cern.ch/jira/browse/SPI-2408) as an example.
* [ ] Make a MR to use the new release in [ATLAS `StatAnalysis`](https://gitlab.cern.ch/atlas/StatAnalysis).
* [ ] Close the [release GitHub Project board](https://github.com/scikit-hep/pyhf/projects/).
