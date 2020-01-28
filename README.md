<p align="center">
<img src="https://raw.githubusercontent.com/scikit-hep/pyhf/master/docs/_static/img/pyhf-logo.png" alt="pyhf logo" width="320"/>
</p>

# pure-python fitting/limit-setting/interval estimation HistFactory-style

[![GitHub Project](https://img.shields.io/badge/GitHub--blue?style=social&logo=GitHub)](https://github.com/scikit-hep/pyhf)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1169739.svg)](https://doi.org/10.5281/zenodo.1169739)
[![Scikit-HEP](https://img.shields.io/badge/Scikit--HEP-Project-blue?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABoAAAAcCAYAAAB/E6/TAAAACXBIWXMAAAEZAAABGQHyCY1sAAAAGXRFWHRTb2Z0d2FyZQB3d3cuaW5rc2NhcGUub3Jnm+48GgAAA6dJREFUSImdlktonFUUx/930kQ0nYo2JX5NUqSghSq2oIgvcCFaC0EEH2AXUh9FEV1UUaGIC924qY8igo+FQi26sWp8gDS24qOgCOIiKCVomLlnZpxk7Dh2GjMz389F7tgv4zfNJGdzh/M/5/zuud/ch9MKDFgnaUjSnHOu2kuOmb0h6brMMoVHgceBY8ApSVVJ05JOAnXga+BJ4OK0fO/9PZL2AL91AwwBLwLz9GYLwKvAcLtGLpcbMbM5MyuXy+UoDXI14BNFcsABYBy4DLgojDvDZH5PxJaAG4CM937SzCgUCnemQcaB0yFpDngMGFhmefuBh4E/Qt4/tVrtoJlhZq+nJWwHaiH4F2DL2QAp+SPA9wBxHDM7O5svl8vZzqBzgOkAOQGsXwmkbbVabUOj0Wh/1xIw+J+Yy+UuBJ4O4jywdTUQSSoUCgdKpRJxHC+Ees8mxVKr1WoGYf9qId77m80sNrNvgedDvb+A8yQpMzg4OJHJZPoAVavVQ6uBmNmQc+4dSfVWq7Vb0n5JC5KyknZIUiabzdYlqdFoqF6vTxSLxctXwXpNUuSce3RsbOyEc+6kpKNBG5ekjKRLguMTSUNxHE/m8/ntvRK89w9IukvS4SiK3k5Ix8N4aRu0UZIGBgaOAHdIWpfJZI56769fDlIqlTY7515yzlkcx3s65xDGjW1Qf3A0R0ZGJpxzOyX1Oee+MLMd3SDAmjiOD0paK+nB0dHRuY6QhTD2t0EWHJEkRVF0zDl3k6TTkj5OPUIkmdkzwLWSXomi6POUkNFkZxlJM8GxrR0RRdEPzrkbnXOzwHve+/s7IFc55/ZJmmq1Wvu6NH1FGGfaS3B3YrMuOTJKpdJmM5sO+2OvJBWLxUEz+9XM5vP5/DalWDhpqqHu7rYzmzhI96Ys0aZQmEKh8IKZvRV+P9GlEwEPhXoNYCgpvByE2SXCmc6GzeyncCLjvZ8EUi9N4HygEOq92SmuB/4M4pdAf2eBmZmZC7z3lQDb1AXSB3wW6vwNpF54twOtEPQBsLYzplgsfmhmpHUDnAscCvkxsCttMu3gpzhjPwNLNq2ZHU4DsXgr/5jIfa4rJJF0H0vfCp8C9wLDSRCwAdgFfBQ6gMW3wyPLQhKwK4Gv+L81m80mwKkU7Tvgmp4hHcBbgXcTf5ROqwLvA7cBblWQDmA/sLVSqXxTqVQAbmHxJXTWh0vS1vQS5JxrSJoys3JwHXHOxSuZbE+ghE1J2rJSiCT9CxJT5EBIY81lAAAAAElFTkSuQmCC)](https://scikit-hep.org/)

[![GitHub Actions Status: CI](https://github.com/scikit-hep/pyhf/workflows/CI/CD/badge.svg)](https://github.com/scikit-hep/pyhf/actions?query=workflow%3ACI%2FCD+branch%3Amaster)
[![GitHub Actions Status: Publish](https://github.com/scikit-hep/pyhf/workflows/publish%20distributions/badge.svg)](https://github.com/scikit-hep/pyhf/actions?query=workflow%3A%22publish+distributions%22+branch%3Amaster)
[![Docker Automated](https://img.shields.io/docker/automated/pyhf/pyhf.svg)](https://hub.docker.com/r/pyhf/pyhf/)
[![Code Coverage](https://codecov.io/gh/scikit-hep/pyhf/graph/badge.svg?branch=master)](https://codecov.io/gh/scikit-hep/pyhf?branch=master)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/scikit-hep/pyhf.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/scikit-hep/pyhf/latest/files/)
[![CodeFactor](https://www.codefactor.io/repository/github/scikit-hep/pyhf/badge)](https://www.codefactor.io/repository/github/scikit-hep/pyhf)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[![Docs](https://img.shields.io/badge/docs-master-blue.svg)](https://scikit-hep.github.io/pyhf)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/scikit-hep/pyhf/master?filepath=docs%2Fexamples%2Fnotebooks%2Fbinderexample%2FStatisticalAnalysis.ipynb)

[![PyPI version](https://badge.fury.io/py/pyhf.svg)](https://badge.fury.io/py/pyhf)
[![Supported Python versionss](https://img.shields.io/pypi/pyversions/pyhf.svg)](https://pypi.org/project/pyhf/)
[![Docker Stars](https://img.shields.io/docker/stars/pyhf/pyhf.svg)](https://hub.docker.com/r/pyhf/pyhf/)
[![Docker Pulls](https://img.shields.io/docker/pulls/pyhf/pyhf.svg)](https://hub.docker.com/r/pyhf/pyhf/)

The HistFactory p.d.f. template [[CERN-OPEN-2012-016](https://cds.cern.ch/record/1456844)] is per-se independent of its implementation in ROOT and sometimes, it's useful to be able to run statistical analysis outside of ROOT, RooFit, RooStats framework.


This repo is a pure-python implementation of that statistical model for multi-bin histogram-based analysis and its interval estimation is based on the asymptotic formulas of "Asymptotic formulae for likelihood-based tests of new physics" [[arXiv:1007.1727](https://arxiv.org/abs/1007.1727)]. The aim is also to support modern computational graph libraries such as PyTorch and TensorFlow in order to make use of features such as autodifferentiation and GPU acceleration.

## Hello World

```python
>>> import pyhf
>>> pdf = pyhf.simplemodels.hepdata_like(signal_data=[12.0, 11.0], bkg_data=[50.0, 52.0], bkg_uncerts=[3.0, 7.0])
>>> CLs_obs, CLs_exp = pyhf.infer.hypotest(1.0, [51, 48] + pdf.config.auxdata, pdf, return_expected=True)
>>> print('Observed: {}, Expected: {}'.format(CLs_obs, CLs_exp))
Observed: [0.05290116], Expected: [0.06445521]

```

## What does it support

Implemented variations:
- [x] HistoSys
- [x] OverallSys
- [x] ShapeSys
- [x] NormFactor
- [x] Multiple Channels
- [x] Import from XML + ROOT via [uproot](https://github.com/scikit-hep/uproot)
- [x] ShapeFactor
- [x] StatError
- [x] Lumi Uncertainty

Computational Backends:
- [x] NumPy
- [x] PyTorch
- [x] TensorFlow
- [x] JAX

Available Optimizers

| NumPy                    | Tensorflow                 | PyTorch                    |
| :----------------------- | :------------------------- | :------------------------- |
| SLSQP (`scipy.optimize`) | Newton's Method (autodiff) | Newton's Method (autodiff) |
| MINUIT (`iminuit`)       | .                          | .                          |


## Todo
- [ ] StatConfig
- [ ] Non-asymptotic calculators

results obtained from this package are validated against output computed from HistFactory workspaces

## A one bin example

```python
nobs = 55, b = 50, db = 7, nom_sig = 10.
```

<img src="https://raw.githubusercontent.com/scikit-hep/pyhf/master/docs/_static/img/manual_1bin_55_50_7.png" alt="manual" width="500"/>
<img src="https://raw.githubusercontent.com/scikit-hep/pyhf/master/docs/_static/img/hfh_1bin_55_50_7.png" alt="manual" width="500"/>


## A two bin example

```python
bin 1: nobs = 100, b = 100, db = 15., nom_sig = 30.
bin 2: nobs = 145, b = 150, db = 20., nom_sig = 45.
```

<img src="https://raw.githubusercontent.com/scikit-hep/pyhf/master/docs/_static/img/manual_2_bin_100.0_145.0_100.0_150.0_15.0_20.0_30.0_45.0.png" alt="manual" width="500"/>
<img src="https://raw.githubusercontent.com/scikit-hep/pyhf/master/docs/_static/img/hfh_2_bin_100.0_145.0_100.0_150.0_15.0_20.0_30.0_45.0.png" alt="manual" width="500"/>

## Installation

To install `pyhf` from PyPI with the NumPy backend run
```bash
python -m pip install pyhf
```

and to install `pyhf` with all additional backends run
```bash
python -m pip install pyhf[backends]
```
or a subset of the options.

To uninstall run
```bash
python -m pip uninstall pyhf
```

## Authors

Please check the [contribution statistics for a list of contributors](https://github.com/scikit-hep/pyhf/graphs/contributors)
