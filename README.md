# pure-python fitting/limit-setting/interval estimation HistFactory-style

[![Build Status](https://travis-ci.org/lukasheinrich/pyhf.svg?branch=master)](https://travis-ci.org/lukasheinrich/pyhf)
[![Coverage Status](https://coveralls.io/repos/github/lukasheinrich/pyhf/badge.svg?branch=master)](https://coveralls.io/github/lukasheinrich/pyhf?branch=master)
[![PyPI version](https://badge.fury.io/py/pyhf.svg)](https://badge.fury.io/py/pyhf)
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/lukasheinrich/pyhf/master)

The HistFactory p.d.f. template is per-se independent of its implementation in ROOT and sometimes, it's useful to be able to run statistical analysis outside
of ROOT, RooFit, RooStats framework.

This repo has some example code for multi-bin histogram-based analysis based on the asymptotic formulas of "Asymptotic formulae for likelihood-based tests of new physics" [[arxiv:1007.1727](https://arxiv.org/abs/1007.1727)].

## What does it support

Implemented variations
- [x] HistoSys
- [x] OverallSys
- [x] ShapeSys
- [x] NormFactor
- [x] Multiple Channels
- [x] Import from XML + ROOT via `uproot` (https://github.com/scikit-hep/uproot)
- [x] ShapeFactor

## Todo
- [ ] StatError
- [ ] StatConfig
- [ ] Non-asymptotic calculators

results obtained from this package are validated against output computed from HistFactory workspaces

## A one bin example

```
nobs = 55, b = 50, db = 7, nom_sig = 10.
```

<img src="docs/img/manual_1bin_55_50_7.png" alt="manual" width="500"/>
<img src="docs/img/hfh_1bin_55_50_7.png" alt="manual" width="500"/>


## A two bin example

```
bin 1: nobs = 100, b = 100, db = 15., nom_sig = 30.
bin 2: nobs = 145, b = 150, db = 20., nom_sig = 45.
```

<img src="docs/img/manual_2_bin_100.0_145.0_100.0_150.0_15.0_20.0_30.0_45.0.png" alt="manual" width="500"/>
<img src="docs/img/hfh_2_bin_100.0_145.0_100.0_150.0_15.0_20.0_30.0_45.0.png" alt="manual" width="500"/>

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
