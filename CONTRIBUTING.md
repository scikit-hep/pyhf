# Contributing to pyhf

We are happy to accept contributions to `pyhf` via Pull Requests to the GitHub repository and welcome Issues.
To get started fork the repo.

## Issues

Making Issues is very helpful to the project &mdash; they help the dev team form the development roadmap and are where most important discussion takes place.
If you have suggestions, questions that you can't find answers to on the [documentation website](https://scikit-hep.org/pyhf/) or on the [Stack Overflow tag](https://stackoverflow.com/questions/tagged/pyhf), or have found a bug please [open an Issue](https://github.com/scikit-hep/pyhf/issues/new/choose)!

## Pull Requests

## Opening an Issue to Discuss

Unless your Pull Request is an obvious 1 line fix, please first [open an Issue](https://github.com/scikit-hep/pyhf/issues/new/choose) to discuss your PR with the dev team.
The Issue allows for discussion on the usefulness and scope of the PR to be publicly discussed and also allows for the PR to then be focused on the code review.

### Good Examples

If you're looking for some examples of high quality contributed pull requests we recommend you take a look at these:

- PR [#902](https://github.com/scikit-hep/pyhf/pull/902) by Nikolai Hartmann ([@nikoladze](https://github.com/nikoladze))

Many thanks goes out to our contributors!

### Drafts

Unless you are making a single commit pull request please create a draft pull request. Outline the work that will be done in this ongoing pull request. When you are close to being done please tag someone with Approver permissions to follow the pull request.

### Pull Request Procedure

If you would like to make a pull request please:

1. Make a fork of the project.
2. Open an Issue to discuss the planned PR with the project maintainers.
3. Commit your changes to a feature branch on your fork and push to your branch.
4. Start a pull request to let the project maintainers know you're working on it.
5. Test your changes with `pytest`.
6. Update your fork to make sure your changes don't conflict with the current state of the master branch.
7. Make sure that you've added your name to `docs/contributors.rst`.
If you haven't **please** do so by simply appending your name to the bottom of the list.
We are thankful for and value your contributions to `pyhf`, not matter the size.
8. Request your PR be reviewed by the project maintainers.

## Bug Reports

If you have found a bug please report it by filling out the [bug report template](https://github.com/scikit-hep/pyhf/issues/new?template=Bug-Report.md&labels=bug&title=Bug+Report+:+Title+Here).

## Installing the development environment

We recommend first reading the "[Developing](https://scikit-hep.org/pyhf/development.html)" page on the pyhf website and the coming back here.

You can install the development environment (which includes a number of extra) libraries and all others needed to run the tests via `pip`:

```
python -m pip install --ignore-installed -U -e .[complete]
```

To make the PR process much smoother we also strongly recommend that you setup the Git pre-commit hook for [Black](https://github.com/psf/black) by running

```
pre-commit install
```

This will run `black` over your code each time you attempt to make a commit and warn you if there is an error, canceling the commit.

## Running the tests

You can run the unit tests (which should be fast!) via the following command.

```
python -m pytest --ignore=tests/test_notebooks.py
```

Note: This ignores the notebook tests (which are run via [papermill](https://github.com/nteract/papermill) which run somewhat slow.
Make sure to run the complete suite before submitting a PR

```
python -m pytest
```

## Making a pull request

We try to follow [Conventional Commit](https://www.conventionalcommits.org/) for commit messages and PR titles. Since we merge PR's using squash commits, it's fine if the final commit messages (proposed in the PR body) follow this convention.

## Generating Reference Visuals

New baseline visuals can be generated using this command:

```
python -m pytest tests/contrib/test_viz.py --mpl-generate-path=tests/contrib/baseline
```
