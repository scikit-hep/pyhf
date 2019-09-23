# Contributing to pyhf

We are happy to accept contributions to `pyhf` via Pull Requests to the GitHub repo. To get started fork the repo.

## Pull Requests

### WIP

Unless you are making a single commit pull request please create a WIP pull request. Outline the work that will be done in this ongoing pull request. When you are close to being done please assign someone with Approver permissions to follow the pull request.

### Pull Requests Procedure

If you would like to make a pull request please:

1. Make a fork of the project
2. Start a pull request to let the project maintainers know you're working on it
3. Commit your changes to the fork and push your branch
4. Test your changes with `pytest`
5. Update your fork to make sure your changes don't conflict with the current state of the master branch
6. Request your changes be accepted

## Bug Reports

If you have found a bug please report it by filling out the [bug report template](https://github.com/diana-hep/pyhf/issues/new?template=Bug-Report.md&labels=bug&title=Bug+Report+:+Title+Here).

## Installing the development environment

We recommend first reading the "[Developing](https://diana-hep.org/pyhf/development.html)" page on the pyhf website and the coming back here.

You can install the development environment (which includes a number of extra) libraries and all others needed to run the tests via `pip`:

```
pip install --ignore-installed -U -e .[complete]
```

To make the PR process much smoother we also strongly recommend that you setup the Git pre-commit hook for [Black](https://github.com/psf/black) by running

```
pre-commit install
```

This will run `black` over your code each time you attempt to make a commit and warn you if there is an error, canceling the commit.

## Running the tests

You can run the unit tests (which should be fast!) via the following command.

```
pytest --ignore=tests/test_notebooks.py
```

Note: This ignores the notebook tests (which are run via [papermill](https://github.com/nteract/papermill) which run somewhat slow.
Make sure to run the complete suite before submitting a PR

```
pytest
```
