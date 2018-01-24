# histfactory_validation

The HistFactory p.d.f. template is per-se independent of its implementation in ROOT and sometimes, it's useful to be able to run statistical analysis outside
of ROOT, RooFit, RooStats framework.

This repo has some example code for multi-bin histogram-based analysis based on the asymptotic formulas of arxiv:1007.1727 and  

So far it only implements a simple model of

* one signal historgam with a NormFactor
* one background histogram with a ShapeSys (uncorrelated bin-by-bin uncertainties)

and validates it against output computed from HistFactory workspaces

## A one bin example

```
nobs = 55, b = 50, db = 7, nom_sig = 10.
```

<img src="img/manual_1bin_55_50_7.png" alt="manual" width="500"/>
<img src="img/hfh_1bin_55_50_7.png" alt="manual" width="500"/>


## A two bin example

```
bin 1: nobs = 100, b = 100, db = 15., nom_sig = 30.
bin 2: nobs = 145, b = 150, db = 20., nom_sig = 45.
```

<img src="img/manual_2_bin_100.0_145.0_100.0_150.0_15.0_20.0_30.0_45.0.png" alt="manual" width="500"/>
<img src="img/hfh_2_bin_100.0_145.0_100.0_150.0_15.0_20.0_30.0_45.0.png" alt="manual" width="500"/>
