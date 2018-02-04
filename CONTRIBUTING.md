# Contributing to pyhf

We are happy to accept contributions to `pyhf` via Pull Requests to the GitHub repo. To get started fork the repo.

## Installing the development environment

You can install the development environment (which includes a number of extra) libraries via `pip`:

```
pip install -e.[develop]
```

## Running the tests

You can run the unit tests (which should be fast!) via the following command.

```
pytest --ignore=tests/test_notebooks.py
```

Note: This ignores the notebook tests (which are run via [papermill](https://github.com/nteract/papermill) ) which run somewhat slow.
Make sure to run the complete suite before submitting a PR

```
pytest
```
