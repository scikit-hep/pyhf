.. image:: https://raw.githubusercontent.com/scikit-hep/pyhf/master/docs/_static/img/pyhf-logo-small.png
   :alt: pyhf logo
   :width: 320
   :align: center

pure-python fitting/limit-setting/interval estimation HistFactory-style
=======================================================================

|GitHub Project| |DOI| |JOSS DOI| |Scikit-HEP| |NSF Award Number|

|GitHub Actions Status: CI| |GitHub Actions Status: Docs| |GitHub Actions Status: Publish|
|Docker Automated| |Code Coverage| |CodeFactor| |pre-commit.ci Status| |Code style: black|

|Docs from latest| |Docs from master| |Binder|

|PyPI version| |Conda-forge version| |Supported Python versions| |Docker Stars| |Docker
Pulls|

The HistFactory p.d.f. template
[`CERN-OPEN-2012-016 <https://cds.cern.ch/record/1456844>`__] is per-se
independent of its implementation in ROOT and sometimes, it’s useful to
be able to run statistical analysis outside of ROOT, RooFit, RooStats
framework.

This repo is a pure-python implementation of that statistical model for
multi-bin histogram-based analysis and its interval estimation is based
on the asymptotic formulas of “Asymptotic formulae for likelihood-based
tests of new physics”
[`arXiv:1007.1727 <https://arxiv.org/abs/1007.1727>`__]. The aim is also
to support modern computational graph libraries such as PyTorch and
TensorFlow in order to make use of features such as autodifferentiation
and GPU acceleration.

Hello World
-----------

This is how you use the ``pyhf`` Python API to build a statistical model and run basic inference:

.. code:: python

   >>> import pyhf
   >>> model = pyhf.simplemodels.hepdata_like(signal_data=[12.0, 11.0], bkg_data=[50.0, 52.0], bkg_uncerts=[3.0, 7.0])
   >>> data = [51, 48] + model.config.auxdata
   >>> test_mu = 1.0
   >>> CLs_obs, CLs_exp = pyhf.infer.hypotest(test_mu, data, model, test_stat="qtilde", return_expected=True)
   >>> print(f"Observed: {CLs_obs}, Expected: {CLs_exp}")
   Observed: 0.05251510242427374, Expected: 0.06445366938143944

Alternatively the statistical model and observational data can be read from its serialized JSON representation (see next section).

.. code:: python

   >>> import pyhf
   >>> import requests
   >>> wspace = pyhf.Workspace(requests.get('https://git.io/JJYDE').json())
   >>> model = wspace.model()
   >>> data = wspace.data(model)
   >>> test_mu = 1.0
   >>> CLs_obs, CLs_exp = pyhf.infer.hypotest(test_mu, data, model, test_stat="qtilde", return_expected=True)
   >>> print(f"Observed: {CLs_obs}, Expected: {CLs_exp}")
   Observed: 0.3599843354244575, Expected: 0.3599843354244575


Finally, you can also use the command line interface that ``pyhf`` provides

.. code:: bash

   $ cat << EOF  | tee likelihood.json | pyhf cls
   {
       "channels": [
           { "name": "singlechannel",
             "samples": [
               { "name": "signal",
                 "data": [12.0, 11.0],
                 "modifiers": [ { "name": "mu", "type": "normfactor", "data": null} ]
               },
               { "name": "background",
                 "data": [50.0, 52.0],
                 "modifiers": [ {"name": "uncorr_bkguncrt", "type": "shapesys", "data": [3.0, 7.0]} ]
               }
             ]
           }
       ],
       "observations": [
           { "name": "singlechannel", "data": [51.0, 48.0] }
       ],
       "measurements": [
           { "name": "Measurement", "config": {"poi": "mu", "parameters": []} }
       ],
       "version": "1.0.0"
   }
   EOF

which should produce the following JSON output:

.. code:: json

   {
      "CLs_exp": [
         0.0026062609501074576,
         0.01382005356161206,
         0.06445320535890459,
         0.23525643861460702,
         0.573036205919389
      ],
      "CLs_obs": 0.05251497423736956
   }

What does it support
--------------------

Implemented variations:
  - ☑ HistoSys
  - ☑ OverallSys
  - ☑ ShapeSys
  - ☑ NormFactor
  - ☑ Multiple Channels
  - ☑ Import from XML + ROOT via `uproot <https://github.com/scikit-hep/uproot>`__
  - ☑ ShapeFactor
  - ☑ StatError
  - ☑ Lumi Uncertainty
  - ☑ Non-asymptotic calculators

Computational Backends:
  - ☑ NumPy
  - ☑ PyTorch
  - ☑ TensorFlow
  - ☑ JAX

Optimizers:
  - ☑ SciPy (``scipy.optimize``)
  - ☑ MINUIT (``iminuit``)

All backends can be used in combination with all optimizers.
Custom user backends and optimizers can be used as well.

Todo
----

-  ☐ StatConfig

results obtained from this package are validated against output computed
from HistFactory workspaces

A one bin example
-----------------

.. code:: python

   import pyhf
   import numpy as np
   import matplotlib.pyplot as plt
   import pyhf.contrib.viz.brazil

   pyhf.set_backend("numpy")
   model = pyhf.simplemodels.hepdata_like(
       signal_data=[10.0], bkg_data=[50.0], bkg_uncerts=[7.0]
   )
   data = [55.0] + model.config.auxdata

   poi_vals = np.linspace(0, 5, 41)
   results = [
       pyhf.infer.hypotest(
           test_poi, data, model, test_stat="qtilde", return_expected_set=True
       )
       for test_poi in poi_vals
   ]

   fig, ax = plt.subplots()
   fig.set_size_inches(7, 5)
   ax.set_xlabel(r"$\mu$ (POI)")
   ax.set_ylabel(r"$\mathrm{CL}_{s}$")
   pyhf.contrib.viz.brazil.plot_results(ax, poi_vals, results)
   fig.show()

**pyhf**

.. image:: https://raw.githubusercontent.com/scikit-hep/pyhf/master/docs/_static/img/README_1bin_example.png
   :alt: manual
   :width: 500
   :align: center

**ROOT**

.. image:: https://raw.githubusercontent.com/scikit-hep/pyhf/master/docs/_static/img/hfh_1bin_55_50_7.png
   :alt: manual
   :width: 500
   :align: center

A two bin example
-----------------

.. code:: python

   import pyhf
   import numpy as np
   import matplotlib.pyplot as plt
   import pyhf.contrib.viz.brazil

   pyhf.set_backend("numpy")
   model = pyhf.simplemodels.hepdata_like(
       signal_data=[30.0, 45.0], bkg_data=[100.0, 150.0], bkg_uncerts=[15.0, 20.0]
   )
   data = [100.0, 145.0] + model.config.auxdata

   poi_vals = np.linspace(0, 5, 41)
   results = [
       pyhf.infer.hypotest(
           test_poi, data, model, test_stat="qtilde", return_expected_set=True
       )
       for test_poi in poi_vals
   ]

   fig, ax = plt.subplots()
   fig.set_size_inches(7, 5)
   ax.set_xlabel(r"$\mu$ (POI)")
   ax.set_ylabel(r"$\mathrm{CL}_{s}$")
   pyhf.contrib.viz.brazil.plot_results(ax, poi_vals, results)
   fig.show()


**pyhf**

.. image:: https://raw.githubusercontent.com/scikit-hep/pyhf/master/docs/_static/img/README_2bin_example.png
   :alt: manual
   :width: 500
   :align: center

**ROOT**

.. image:: https://raw.githubusercontent.com/scikit-hep/pyhf/master/docs/_static/img/hfh_2_bin_100.0_145.0_100.0_150.0_15.0_20.0_30.0_45.0.png
   :alt: manual
   :width: 500
   :align: center

Installation
------------

To install ``pyhf`` from PyPI with the NumPy backend run

.. code:: bash

   python -m pip install pyhf

and to install ``pyhf`` with all additional backends run

.. code:: bash

   python -m pip install pyhf[backends]

or a subset of the options.

To uninstall run

.. code:: bash

   python -m pip uninstall pyhf

Questions
---------

If you have a question about the use of ``pyhf`` not covered in `the
documentation <https://pyhf.readthedocs.io/>`__, please ask a question
on the `GitHub Discussions <https://github.com/scikit-hep/pyhf/discussions>`__.

If you believe you have found a bug in ``pyhf``, please report it in the
`GitHub
Issues <https://github.com/scikit-hep/pyhf/issues/new?template=Bug-Report.md&labels=bug&title=Bug+Report+:+Title+Here>`__.
If you're interested in getting updates from the ``pyhf`` dev team and release
announcements you can join the |pyhf-announcements mailing list|_.

.. |pyhf-announcements mailing list| replace:: ``pyhf-announcements`` mailing list
.. _pyhf-announcements mailing list: https://groups.google.com/group/pyhf-announcements/subscribe

Citation
--------

As noted in `Use and Citations <https://scikit-hep.org/pyhf/citations.html>`__,
the preferred BibTeX entry for citation of ``pyhf`` includes both the
`Zenodo <https://zenodo.org/>`__ archive and the
`JOSS <https://joss.theoj.org/>`__ paper:

.. code:: bibtex

   @software{pyhf,
     author = "{Heinrich, Lukas and Feickert, Matthew and Stark, Giordon}",
     title = "{pyhf: v0.5.4}",
     version = {0.5.4},
     doi = {10.5281/zenodo.1169739},
     url = {https://github.com/scikit-hep/pyhf},
   }

   @article{pyhf_joss,
     doi = {10.21105/joss.02823},
     url = {https://doi.org/10.21105/joss.02823},
     year = {2021},
     publisher = {The Open Journal},
     volume = {6},
     number = {58},
     pages = {2823},
     author = {Lukas Heinrich and Matthew Feickert and Giordon Stark and Kyle Cranmer},
     title = {pyhf: pure-Python implementation of HistFactory statistical models},
     journal = {Journal of Open Source Software}
   }

Authors
-------

``pyhf`` is openly developed by Lukas Heinrich, Matthew Feickert, and Giordon Stark.

Please check the `contribution statistics for a list of
contributors <https://github.com/scikit-hep/pyhf/graphs/contributors>`__.

Milestones
----------

- 2020-07-28: 1000 GitHub issues and pull requests. (See PR `#1000 <https://github.com/scikit-hep/pyhf/pull/1000>`__)

Acknowledgements
----------------

Matthew Feickert has received support to work on ``pyhf`` provided by NSF
cooperative agreement `OAC-1836650 <https://www.nsf.gov/awardsearch/showAward?AWD_ID=1836650>`__ (IRIS-HEP)
and grant `OAC-1450377 <https://www.nsf.gov/awardsearch/showAward?AWD_ID=1450377>`__ (DIANA/HEP).

.. |GitHub Project| image:: https://img.shields.io/badge/GitHub--blue?style=social&logo=GitHub
   :target: https://github.com/scikit-hep/pyhf
.. |DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.1169739.svg
   :target: https://doi.org/10.5281/zenodo.1169739
.. |JOSS DOI| image:: https://joss.theoj.org/papers/10.21105/joss.02823/status.svg
   :target: https://doi.org/10.21105/joss.02823
.. |Scikit-HEP| image:: https://scikit-hep.org/assets/images/Scikit--HEP-Project-blue.svg
   :target: https://scikit-hep.org/
.. |NSF Award Number| image:: https://img.shields.io/badge/NSF-1836650-blue.svg
   :target: https://nsf.gov/awardsearch/showAward?AWD_ID=1836650
.. |GitHub Actions Status: CI| image:: https://github.com/scikit-hep/pyhf/workflows/CI/CD/badge.svg?branch=master
   :target: https://github.com/scikit-hep/pyhf/actions?query=workflow%3ACI%2FCD+branch%3Amaster
.. |GitHub Actions Status: Docs| image:: https://github.com/scikit-hep/pyhf/workflows/Docs/badge.svg?branch=master
   :target: https://github.com/scikit-hep/pyhf/actions?query=workflow%3ADocs+branch%3Amaster
.. |GitHub Actions Status: Publish| image:: https://github.com/scikit-hep/pyhf/workflows/publish%20distributions/badge.svg?branch=master
   :target: https://github.com/scikit-hep/pyhf/actions?query=workflow%3A%22publish+distributions%22+branch%3Amaster
.. |Docker Automated| image:: https://img.shields.io/docker/automated/pyhf/pyhf.svg
   :target: https://hub.docker.com/r/pyhf/pyhf/
.. |Code Coverage| image:: https://codecov.io/gh/scikit-hep/pyhf/graph/badge.svg?branch=master
   :target: https://codecov.io/gh/scikit-hep/pyhf?branch=master
.. |CodeFactor| image:: https://www.codefactor.io/repository/github/scikit-hep/pyhf/badge
   :target: https://www.codefactor.io/repository/github/scikit-hep/pyhf
.. |pre-commit.ci Status| image:: https://results.pre-commit.ci/badge/github/scikit-hep/pyhf/master.svg
  :target: https://results.pre-commit.ci/latest/github/scikit-hep/pyhf/master
  :alt: pre-commit.ci status
.. |Code style: black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
.. |Docs from latest| image:: https://img.shields.io/badge/docs-v0.5.4-blue.svg
   :target: https://pyhf.readthedocs.io/
.. |Docs from master| image:: https://img.shields.io/badge/docs-master-blue.svg
   :target: https://scikit-hep.github.io/pyhf
.. |Binder| image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/scikit-hep/pyhf/master?filepath=docs%2Fexamples%2Fnotebooks%2Fbinderexample%2FStatisticalAnalysis.ipynb
.. |PyPI version| image:: https://badge.fury.io/py/pyhf.svg
   :target: https://badge.fury.io/py/pyhf
.. |Conda-forge version| image:: https://img.shields.io/conda/vn/conda-forge/pyhf.svg
   :target: https://github.com/conda-forge/pyhf-feedstock
.. |Supported Python versions| image:: https://img.shields.io/pypi/pyversions/pyhf.svg
   :target: https://pypi.org/project/pyhf/
.. |Docker Stars| image:: https://img.shields.io/docker/stars/pyhf/pyhf.svg
   :target: https://hub.docker.com/r/pyhf/pyhf/
.. |Docker Pulls| image:: https://img.shields.io/docker/pulls/pyhf/pyhf.svg
   :target: https://hub.docker.com/r/pyhf/pyhf/
