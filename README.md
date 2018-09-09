# pure-python fitting/limit-setting/interval estimation HistFactory-style

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1169739.svg)](https://doi.org/10.5281/zenodo.1169739)

[![Build Status](https://travis-ci.org/diana-hep/pyhf.svg?branch=master)](https://travis-ci.org/diana-hep/pyhf)
[![Docker Automated](https://img.shields.io/docker/automated/pyhf/pyhf.svg)](https://hub.docker.com/r/pyhf/pyhf/)
[![Coverage Status](https://coveralls.io/repos/github/diana-hep/pyhf/badge.svg?branch=master)](https://coveralls.io/github/diana-hep/pyhf?branch=master) [![Code Health](https://landscape.io/github/diana-hep/pyhf/master/landscape.svg?style=flat)](https://landscape.io/github/diana-hep/pyhf/master)

[![Docs](https://img.shields.io/badge/docs-master-blue.svg)](https://diana-hep.github.io/pyhf)
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/diana-hep/pyhf/master?filepath=docs%2Fexamples%2Fnotebooks%2Fbinderexample%2FStatisticalAnalysis.ipynb)

[![PyPI version](https://badge.fury.io/py/pyhf.svg)](https://badge.fury.io/py/pyhf)
[![Docker Stars](https://img.shields.io/docker/stars/pyhf/pyhf.svg)](https://hub.docker.com/r/pyhf/pyhf/)
[![Docker Pulls](https://img.shields.io/docker/pulls/pyhf/pyhf.svg)](https://hub.docker.com/r/pyhf/pyhf/)

The HistFactory p.d.f. template [[CERN-OPEN-2012-016](https://cds.cern.ch/record/1456844)] is per-se independent of its implementation in ROOT and sometimes, it's useful to be able to run statistical analysis outside
of ROOT, RooFit, RooStats framework.

This repo is a pure-python implementation of that statistical model for multi-bin histogram-based analysis and its interval estimation is based on the asymptotic formulas of "Asymptotic formulae for likelihood-based tests of new physics" [[arxiv:1007.1727](https://arxiv.org/abs/1007.1727)]. The aim is also to support modern computational graph libraries such as PyTorch and TensorFlow in order to make use of features such as autodifferentiation and GPU acceleration.

## Hello World

```python
>>> import pyhf
>>> import pyhf.simplemodels
>>> import pyhf.utils
>>> pdf = pyhf.simplemodels.hepdata_like(signal_data=[12.,11.], bkg_data=[50.,52.], bkg_uncerts=[3.,7.])
>>> *_, CLs_obs,CLs_exp = pyhf.utils.runOnePoint(1.0, [51, 48] + pdf.config.auxdata, pdf)
>>> print('Observed: {} Expected: {}'.format(CLs_obs, CLs_exp[2]))
Observed: [0.05290116] Expected: [0.06445521]

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

Computational Backends:
- [x] NumPy
- [x] PyTorch
- [x] TensorFlow
- [x] MXNet

Available Optimizers

| NumPy        |Tensorflow           |PyTorch  | MxNet|
|:------------- |:-------------:|:-----|:-|
| SLSQP (`scipy.optimize`)    | Newton's Method (autodiff)| Newton's Method (autodiff) | N/A |
| MINUIT (`iminuit`)      |       |    | |


## Todo
- [ ] Lumi Uncertainty
- [ ] StatConfig
- [ ] Non-asymptotic calculators

results obtained from this package are validated against output computed from HistFactory workspaces

## A one bin example

```
nobs = 55, b = 50, db = 7, nom_sig = 10.
```

<img src="docs/_static/img/manual_1bin_55_50_7.png" alt="manual" width="500"/>
<img src="docs/_static/img/hfh_1bin_55_50_7.png" alt="manual" width="500"/>


## A two bin example

```
bin 1: nobs = 100, b = 100, db = 15., nom_sig = 30.
bin 2: nobs = 145, b = 150, db = 20., nom_sig = 45.
```

<img src="docs/_static/img/manual_2_bin_100.0_145.0_100.0_150.0_15.0_20.0_30.0_45.0.png" alt="manual" width="500"/>
<img src="docs/_static/img/hfh_2_bin_100.0_145.0_100.0_150.0_15.0_20.0_30.0_45.0.png" alt="manual" width="500"/>

## Installation

To install `pyhf` run
```bash
python setup.py install
```

To uninstall run
```bash
pip uninstall pyhf
```

## Authors

Please check the [contribution statistics for a list of contributors](https://github.com/lukasheinrich/pyhf/graphs/contributors)
