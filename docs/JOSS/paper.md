---
title: 'pyhf: pure-Python implementation of HistFactory statistical models'
tags:
  - Python
  - physics
  - high energy physics
  - statistical modeling
  - fitting
  - auto-differentiation
authors:
  - name: Lukas Heinrich
    orcid: 0000-0002-4048-7584
    affiliation: 1
  - name: Matthew Feickert^[Corresponding author.]
    orcid: 0000-0003-4124-7862
    affiliation: 2
  - name: Giordon Stark
    orcid: 0000-0001-6616-3433
    affiliation: 3
affiliations:
 - name: CERN
   index: 1
 - name: University of Illinois at Urbana-Champaign
   index: 2
 - name: SCIPP, University of California, Santa Cruz
   index: 3
date: 5 October 2020
bibliography: paper.bib
---

# Summary

Measurements in High Energy Physics (HEP) rely on determining the compatibility of observed collision events with theoretical predictions.
The relationship between them is often formalised in a statistical model $f(\mathbf{x}|\mathbf{\phi})$ describing the probability of data $\mathbf{x}$ given model parameters $\mathbf{\phi}$.
Given observed data, the likelihood $\mathcal{L}(\mathbf{\phi})$ then serves as the basis to test hypotheses on the parameters $\mathbf{\phi}$.
For measurements based on binned data (histograms), the `HistFactory` family of statistical models [@Cranmer:1456844] has been widely used in both Standard Model measurements [@HIGG-2013-02] as well as searches for new physics [@ATLAS-CONF-2018-041].
`pyhf` is the first pure-Python implementation of the `HistFactory` model specification, and implements a declarative, plain-text format for describing `HistFactory`-based likelihoods that is targeted for reinterpretation and long-term preservation in analysis data repositories such as HEPData [@Maguire:2017ypu].

Through adoption of open source "tensor" computational Python libraries, `pyhf` decreases the abstractions between a physicist performing an analysis and the statistical modeling without sacrificing computational speed.
By taking advantage of tensor calculations, `pyhf` outperforms the traditional `C++` implementation of `HistFactory` on data from real LHC analyses.
`pyhf`'s default computational backend is built from NumPy and SciPy, and supports TensorFlow, PyTorch, and JAX as alternative backend choices.
These alternative backends support hardware acceleration on GPUs, and in the case of JAX JIT compilation, as well as auto-differentiation allowing for calculating the full gradient of the likelihood function &mdash; all contributing to speeding up fits.

## Impact on Physics

In addition to enabling the first publication of full likelihoods by an LHC experiment [@ATL-PHYS-PUB-2019-029], `pyhf` has been used by the `SModels` library to improve the reinterpretation of results of searches for new physics at LHC experiments [@Abdallah:2020pec], [@Khosa:2020zar], [@Alguero:2020grj].

## Future work

Future development aims to provide support limit setting through pseudoexperiment generation in the regimes in which asymptotic approximations [@Cowan:2010js] are no longer valid.
Further improvements to the performance of the library as well as API refinement are also planned.

# Acknowledgements

We would like to thank Kyle Cranmer for discussions on `HistFactory` and guidance in the early stages of `pyhf` development, and thank our fellow developers in the Scikit-HEP community for their continued support and feedback.
Matthew Feickert has received support to work on `pyhf` provided by NSF cooperative agreement OAC-1836650 (IRIS-HEP) and grant OAC-1450377 (DIANA/HEP).

# References
