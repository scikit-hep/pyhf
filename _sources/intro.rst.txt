.. _sec:intro:

Introduction
============

Measurements in High Energy Physics (HEP) rely on determining the
compatibility of observed collision events with theoretical predictions.
The relationship between them is often formalised in a statistical *model*
:math:`f(\bm{x}|\fullset)` describing the probability of data
:math:`\bm{x}` given model parameters :math:`\fullset`. Given observed
data, the *likelihood* :math:`\mathcal{L}(\fullset)` then serves as the basis to test
hypotheses on the parameters \ :math:`\fullset`. For measurements based
on binned data (*histograms*), the :math:`\HiFa{}` family of statistical models has been widely used
in both Standard Model measurements :cite:`intro-HIGG-2013-02` as
well as searches for new
physics :cite:`intro-ATLAS-CONF-2018-041`. In this package, a
declarative, plain-text format for describing :math:`\HiFa{}`-based likelihoods is
presented that is targeted for reinterpretation and long-term
preservation in analysis data repositories such as
HEPData :cite:`intro-Maguire:2017ypu`.

HistFactory
-----------

Statistical models described using :math:`\HiFa{}` :cite:`intro-Cranmer:1456844`
center around the simultaneous measurement of disjoint binned
distributions (*channels*) observed as event counts :math:`\channelcounts`. For
each channel, the overall expected event rate [1]_ is the sum over a
number of physics processes (*samples*). The sample rates may be subject to
parametrised variations, both to express the effect of *free parameters*
:math:`\freeset` [2]_ and to account for systematic uncertainties as a
function of *constrained parameters* :math:`\constrset`. The degree to which the latter can cause
a deviation of the expected event rates from the nominal rates is
limited by *constraint terms*. In a frequentist framework these constraint terms can be
viewed as *auxiliary measurements* with additional global observable data :math:`\auxdata`, which
paired with the channel data :math:`\channelcounts` completes the
observation :math:`\bm{x} =
(\channelcounts,\auxdata)`. In addition to the partition of the full
parameter set into free and constrained parameters :math:`\fullset =
(\freeset,\constrset)`, a separate partition :math:`\fullset =
(\poiset,\nuisset)` will be useful in the context of hypothesis testing,
where a subset of the parameters are declared *parameters of interest* :math:`\poiset` and the
remaining ones as *nuisance parameters* :math:`\nuisset`.

.. math::
    :label: eqn:parameters_partitions

    f(\bm{x}|\fullset) = f(\bm{x}|\overbrace{\freeset}^{\llap{\text{free}}},\underbrace{\constrset}_{\llap{\text{constrained}}}) = f(\bm{x}|\overbrace{\poiset}^{\rlap{\text{parameters of interest}}},\underbrace{\nuisset}_{\rlap{\text{nuisance parameters}}})

Thus, the overall structure of a :math:`\HiFa{}` probability model is a product of the
analysis-specific model term describing the measurements of the channels
and the analysis-independent set of constraint terms:

.. math::
    :label: eqn:hifa_template

    f(\channelcounts, \auxdata \,|\,\freeset,\constrset) = \underbrace{\color{blue}{\prod_{c\in\mathrm{\,channels}} \prod_{b \in \mathrm{\,bins}_c}\textrm{Pois}\left(n_{cb} \,\middle|\, \nu_{cb}\left(\freeset,\constrset\right)\right)}}_{\substack{\text{Simultaneous measurement}\\%
      \text{of multiple channels}}} \underbrace{\color{red}{\prod_{\singleconstr \in \constrset} c_{\singleconstr}(a_{\singleconstr} |\, \singleconstr)}}_{\substack{\text{constraint terms}\\%
      \text{for }\unicode{x201C}\text{auxiliary measurements}\unicode{x201D}}},

where within a certain integrated luminosity we observe :math:`n_{cb}`
events given the expected rate of events
:math:`\nu_{cb}(\freeset,\constrset)` as a function of unconstrained
parameters :math:`\freeset` and constrained parameters
:math:`\constrset`. The latter has corresponding one-dimensional
constraint terms
:math:`c_\singleconstr(a_\singleconstr|\,\singleconstr)` with auxiliary
data :math:`a_\singleconstr` constraining the parameter
:math:`\singleconstr`. The event rates :math:`\nu_{cb}` are defined as

.. math::
    :label: eqn:sample_rates

    \nu_{cb}\left(\fullset\right) = \sum_{s\in\mathrm{\,samples}} \nu_{scb}\left(\freeset,\constrset\right) = \sum_{s\in\mathrm{\,samples}}\underbrace{\left(\prod_{\kappa\in\,\bm{\kappa}} \kappa_{scb}\left(\freeset,\constrset\right)\right)}_{\text{multiplicative modifiers}}\, \Bigg(\nu_{scb}^0\left(\freeset, \constrset\right) + \underbrace{\sum_{\Delta\in\bm{\Delta}} \Delta_{scb}\left(\freeset,\constrset\right)}_{\text{additive modifiers}}\Bigg)\,.

The total rates are the sum over sample rates :math:`\nu_{csb}`, each
determined from a *nominal rate* :math:`\nu_{scb}^0` and a set of multiplicative and
additive denoted *rate modifiers* :math:`\bm{\kappa}(\fullset)` and
:math:`\bm{\Delta}(\fullset)`. These modifiers are functions of (usually
a single) model parameters. Starting from constant nominal rates, one
can derive the per-bin event rate modification by iterating over all
sample rate modifications as shown in :eq:`eqn:sample_rates`.

As summarised in :ref:`tab:modifiers_and_constraints`, rate modifications
are defined in :math:`\HiFa{}` for bin :math:`b`, sample :math:`s`, channel
:math:`c`.  Each modifier is represented by a parameter :math:`\phi \in
\{\gamma, \alpha, \lambda, \mu\}`.  By convention bin-wise parameters are
denoted with :math:`\gamma` and interpolation parameters with :math:`\alpha`.
The luminosity :math:`\lambda` and scale factors :math:`\mu` affect all bins
equally.  For constrained modifiers, the implied constraint term is given as
well as the necessary input data required to construct it.  :math:`\sigma_b`
corresponds to the relative uncertainty of the event rate, whereas
:math:`\delta_b` is the event rate uncertainty of the sample relative to the
total event rate :math:`\nu_b = \sum_s \nu^0_{sb}`.

Modifiers implementing uncertainties are paired with
a corresponding default constraint term on the parameter limiting the
rate modification. The available modifiers may affect only the total
number of expected events of a sample within a given channel, i.e. only
change its normalisation, while holding the distribution of events
across the bins of a channel, i.e. its “shape”, invariant.
Alternatively, modifiers may change the sample shapes. Here :math:`\HiFa{}` supports
correlated an uncorrelated bin-by-bin shape modifications. In the
former, a single nuisance parameter affects the expected sample rates
within the bins of a given channel, while the latter introduces one
nuisance parameter for each bin, each with their own constraint term.
For the correlated shape and normalisation uncertainties, :math:`\HiFa{}` makes use of
interpolating functions, :math:`f_p` and :math:`g_p`, constructed from a
small number of evaluations of the expected rate at fixed values of the
parameter :math:`\alpha` [3]_. For the remaining modifiers, the
parameter directly affects the rate.

.. _tab:modifiers_and_constraints:

.. table:: Modifiers and Constraints

    ==================== ============================================================================================================= ===================================================================================================== ================================
    Description          Modification                                                                                                  Constraint Term :math:`c_\singleconstr`                                                               Input
    ==================== ============================================================================================================= ===================================================================================================== ================================
    Uncorrelated Shape   :math:`\kappa_{scb}(\gamma_b) = \gamma_b`                                                                     :math:`\prod_b \mathrm{Pois}\left(r_b = \sigma_b^{-2}\middle|\,\rho_b = \sigma_b^{-2}\gamma_b\right)` :math:`\sigma_{b}`
    Correlated Shape     :math:`\Delta_{scb}(\alpha) = f_p\left(\alpha\middle|\,\Delta_{scb,\alpha=-1},\Delta_{scb,\alpha = 1}\right)` :math:`\displaystyle\mathrm{Gaus}\left(a = 0\middle|\,\alpha,\sigma = 1\right)`                       :math:`\Delta_{scb,\alpha=\pm1}`
    Normalisation Unc.   :math:`\kappa_{scb}(\alpha) = g_p\left(\alpha\middle|\,\kappa_{scb,\alpha=-1},\kappa_{scb,\alpha=1}\right)`   :math:`\displaystyle\mathrm{Gaus}\left(a = 0\middle|\,\alpha,\sigma = 1\right)`                       :math:`\kappa_{scb,\alpha=\pm1}`
    MC Stat. Uncertainty :math:`\kappa_{scb}(\gamma_b) = \gamma_b`                                                                     :math:`\prod_b \mathrm{Gaus}\left(a_{\gamma_b} = 1\middle|\,\gamma_b,\delta_b\right)`                 :math:`\delta_b^2 = \sum_s\delta^2_{sb}`
    Luminosity           :math:`\kappa_{scb}(\lambda) = \lambda`                                                                       :math:`\displaystyle\mathrm{Gaus}\left(l = \lambda_0\middle|\,\lambda,\sigma_\lambda\right)`          :math:`\lambda_0,\sigma_\lambda`
    Normalisation        :math:`\kappa_{scb}(\mu_b) = \mu_b`
    Data-driven Shape    :math:`\kappa_{scb}(\gamma_b) = \gamma_b`
    ==================== ============================================================================================================= ===================================================================================================== ================================

Given the likelihood :math:`\mathcal{L}(\fullset)`, constructed from
observed data in all channels and the implied auxiliary data, *measurements* in the
form of point and interval estimates can be defined. The majority of the
parameters are *nuisance parameters* — parameters that are not the main target of the
measurement but are necessary to correctly model the data. A small
subset of the unconstrained parameters may be declared as *parameters of interest* for which
measurements hypothesis tests are performed, e.g. profile likelihood
methods :cite:`intro-Cowan:2010js`. The :ref:`tab:symbol_summary` table provides a summary of all the
notation introduced in this documentation.

.. _tab:symbol_summary:

.. table:: Symbol Notation

    =================================================================== ===============================================================
    Symbol                                                              Name
    =================================================================== ===============================================================
    :math:`f(\bm{x} | \fullset)`                                        model
    :math:`\mathcal{L}(\fullset)`                                       likelihood
    :math:`\bm{x} = \{\channelcounts, \auxdata\}`                       full dataset (including auxiliary data)
    :math:`\channelcounts`                                              channel data (or event counts)
    :math:`\auxdata`                                                    auxiliary data
    :math:`\nu(\fullset)`                                               calculated event rates
    :math:`\fullset = \{\freeset, \constrset\} = \{\poiset, \nuisset\}` all parameters
    :math:`\freeset`                                                    free parameters
    :math:`\constrset`                                                  constrained parameters
    :math:`\poiset`                                                     parameters of interest
    :math:`\nuisset`                                                    nuisance parameters
    :math:`\bm{\kappa}(\fullset)`                                       multiplicative rate modifier
    :math:`\bm{\Delta}(\fullset)`                                       additive rate modifier
    :math:`c_\singleconstr(a_\singleconstr | \singleconstr)`            constraint term for constrained parameter :math:`\singleconstr`
    :math:`\sigma_\singleconstr`                                        relative uncertainty in the constrained parameter
    =================================================================== ===============================================================

Declarative Formats
-------------------

While flexible enough to describe a wide range of LHC measurements, the
design of the :math:`\HiFa{}` specification is sufficiently simple to admit a *declarative format* that fully
encodes the statistical model of the analysis. This format defines the
channels, all associated samples, their parameterised rate modifiers and
implied constraint terms as well as the measurements. Additionally, the
format represents the mathematical model, leaving the implementation of
the likelihood minimisation to be analysis-dependent and/or
language-dependent. Originally XML was chosen as a specification
language to define the structure of the model while introducing a
dependence on :math:`\Root{}` to encode the nominal rates and required input data of the
constraint terms :cite:`intro-Cranmer:1456844`. Using this
specification, a model can be constructed and evaluated within the
:math:`\RooFit{}` framework.

This package introduces an updated form of the specification based on
the ubiquitous plain-text JSON format and its schema-language *JSON Schema*.
Described in more detail in :ref:`sec:likelihood`, this schema fully specifies both structure
and necessary constrained data in a single document and thus is
implementation independent.

Additional Material
-------------------

Footnotes
~~~~~~~~~

.. [1]
   Here rate refers to the number of events expected to be observed
   within a given data-taking interval defined through its integrated
   luminosity. It often appears as the input parameter to the Poisson
   distribution, hence the name “rate”.

.. [2]
   These *free parameters* frequently include the of a given process, i.e. its cross-section
   normalised to a particular reference cross-section such as that expected
   from the Standard Model or a given BSM scenario.

.. [3]
   This is usually constructed from the nominal rate and measurements of the
   event rate at :math:`\alpha=\pm1`, where the value of the modifier at
   :math:`\alpha=\pm1` must be provided and the value at :math:`\alpha=0`
   corresponds to the corresponding identity operation of the modifier, i.e.
   :math:`f_{p}(\alpha=0) = 0` and :math:`g_{p}(\alpha = 0)=1` for additive and
   multiplicative modifiers respectively. See Section 4.1
   in :cite:`intro-Cranmer:1456844`.

Bibliography
~~~~~~~~~~~~

.. bibliography:: bib/docs.bib
   :filter: docname in docnames
   :style: plain
   :keyprefix: intro-
   :labelprefix: intro-
