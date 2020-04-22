Roadmap (2019-2020)
===================

This is the pyhf 2019 into 2020 Roadmap (Issue
`#561 <https://github.com/scikit-hep/pyhf/issues/561>`__).

Overview and Goals
------------------

We will follow loosely Seibert’s `Heirarchy of
Needs <https://twitter.com/FRoscheck/status/1159158552298229763>`__

|Seibert Heirarchy of Needs SciPy 2019| (`Stan
Seibert <https://github.com/seibert>`__, SciPy 2019)

As a general overview that will include:

-  Improvements to docs

   -  Add lots of examples
   -  Add at least 5 well documented case studies

-  Issue cleanup
-  Adding core feature support
-  "pyhf evolution": integration with columnar data analysis systems
-  GPU support and testing
-  Publications

   -  Submit pyhf to JOSS
   -  Submit pyhf to pyOpenSci
   -  Start pyhf paper in 2020

-  Align with IRIS-HEP Analysis Systems NSF milestones

Time scale
----------

The roadmap will be executed over mostly Quarter 3 of 2019 through
Quarter 1 of 2020, with some projects continuing into Quarter 2 of 2020

-  2019-Q3
-  2019-Q4
-  2020-Q1
-  (2020-Q2)

Roadmap
-------

1. **Documentation and Deployment**

   -  |uncheck| Add docstrings to all functions and classes (Issues #38, #349)
      [2019-Q3]
   -  |uncheck| `Greatly revise and expand
      examples <https://github.com/scikit-hep/pyhf/issues?q=is%3Aopen+is%3Aissue+label%3Adocs>`__
      (Issues #168, #202, #212, #325, #342, #349, #367) [2019-Q3 →
      2019-Q4]

      -  |uncheck| Add small case studies with published sbottom likelihood from
         HEPData

   -  |check| Move to `scikit-hep <https://github.com/scikit-hep>`__ GitHub
      organization [2019-Q3]
   -  |uncheck| Develop a release schedule/criteria [2019-Q4]
   -  |check| Automate deployment with [STRIKEOUT:Azure pipeline (talk with
      Henry Schreiner) (Issue #517)] GitHub Actions (Issue #508)
      [2019-Q3]
   -  |uncheck| Finalize logo and add it to website (Issue #453) [2019-Q3 →
      2019-Q4]
   -  |uncheck| Write submission to `JOSS <https://joss.theoj.org/>`__ (Issue
      #502) and write submission to
      `pyOpenSci <https://www.pyopensci.org/>`__ [2019-Q4 → 2020-Q2]
   -  |uncheck| Contribute to `IRIS-HEP Analysis Systems
      Milestones <https://docs.google.com/spreadsheets/d/1VKpHlQWXu_p8AUv5E5H_BzqF_i7hh2Z-Id0XPwNHu8o/edit#gid=1864915304>`__
      "`Initial roadmap for ecosystem
      coherency <https://github.com/iris-hep/project-milestones/issues/8>`__"
      and "`Initial roadmap for high-level cyberinfrastructure
      components of analysis
      system <https://github.com/iris-hep/project-milestones/issues/11>`__"
      [2019-Q4 → 2020-Q2]

2. **Revision and Maintenance**

   -  |check| Add tests using HEPData published sbottom likelihoods (Issue
      #518) [2019-Q3]
   -  |check| Add CI with GitHub Actions and Azure Pipelines (PR #527, Issue
      #517) [2019-Q3]
   -  |uncheck| Investigate rewrite of pytest fixtures to use modern pytest
      (Issue #370) [2019-Q3 → 2019-Q4]
   -  |check| Factorize out the statistical fitting portion into
      ``pyhf.infer`` (PR #531) [2019-Q3 → 2019-Q4]
   -  |uncheck| Bug squashing at large [2019-Q3 → 2020-Q2]

      -  |uncheck| Unexpected use cases (Issues #324, #325, #529)
      -  |uncheck| Computational edge cases (Issues #332, #445)

   -  |uncheck| Make sure that all backends reproduce sbottom results [2019-Q4 →
      2020-Q2]

3. **Development**

   -  |check| Batch support (PR #503) [2019-Q3]
   -  |check| Add ParamViewer support (PR #519) [2019-Q3]
   -  |check| Add setting of NPs constant/fixed (PR #653) [2019-Q3]
   -  |check| Implement pdf as subclass of distributions (PR #551) [2019-Q3]
   -  |check| Add sampling with toys (PR #558) [2019-Q3]
   -  |uncheck| Make general modeling choices (e.g., Issue #293) [2019-Q4 →
      2020-Q1]
   -  |uncheck| Add "discovery" test stats (p0) (PR #520) [2019-Q4 → 2020-Q1]
   -  |uncheck| Add better Model creation [2019-Q4 → 2020-Q1]
   -  |uncheck| Add background model support (Issue #514) [2019-Q4 → 2020-Q1]
   -  |uncheck| Develop interface for the optimizers similar to tensor/backend
      [2019-Q4 → 2020-Q1]
   -  |check| Migrate to TensorFlow v2.0 (PR #541) [2019-Q4]
   -  |check| Drop Python 2.7 support at end of 2019 (Issue #469) [2019-Q4
      (last week of December 2019)]
   -  |uncheck| Finalize public API [2020-Q1]
   -  |uncheck| Integrate pyfitcore/Statisfactory API [2020-Q1]

4. **Research**

   -  |uncheck| Add pyfitcore/Statisfactory integrations (Issue #344, `zfit
      Issue 120 <https://github.com/zfit/zfit/issues/120>`__) [2019-Q4]
   -  |uncheck| Hardware acceleration scaling studies (Issues #93, #301)
      [2019-Q4 → 2020-Q1]
   -  |uncheck| Speedup through Numba (Issue #364) [2019-Q3 → 2019-Q4]
   -  |uncheck| Dask backend (Issue #259) [2019-Q3 → 2020-Q1]
   -  |uncheck| Attempt to use pyhf as fitting tool for full Analysis Systems
      pipeline test in early 2020 [2019-Q4 → 2020-Q1]
   -  |uncheck| pyhf should satisfy `IRIS-HEP Analysis Systems
      Milestone <https://docs.google.com/spreadsheets/d/1VKpHlQWXu_p8AUv5E5H_BzqF_i7hh2Z-Id0XPwNHu8o/edit#gid=1864915304>`__
      "`GPU/accelerator-based implementation of statistical and other
      appropriate
      components <https://github.com/iris-hep/project-milestones/issues/15>`__"
      [2020-Q1 → 2020-Q2] and contributes to "`Benchmarking and
      assessment of prototype analysis system
      components <https://github.com/iris-hep/project-milestones/issues/17>`__"
      [2020-Q3 → 2020-Q4]

Roadmap as Gantt Chart
~~~~~~~~~~~~~~~~~~~~~~

.. figure:: https://user-images.githubusercontent.com/5142394/64583069-53049180-d355-11e9-8b39-8b2a4599e21e.png
   :alt: pyhf_AS_gantt


Presentations During Roadmap Timeline
-------------------------------------

-  |check| `Talk at IRIS-HEP Institute
   Retreat <https://indico.cern.ch/event/840472/contributions/3564386/>`__
   (September 12-13th, 2019)
-  |check| Talk at `PyHEP 2019 <https://indico.cern.ch/event/833895/>`__
   (October 16-18th, 2019)
-  |check| `Talk at CHEP
   2019 <https://indico.cern.ch/event/773049/contributions/3476143/>`__
   (November 4-8th, 2019)
-  |check| `Poster at CHEP
   2019 <https://indico.cern.ch/event/773049/contributions/3476180/>`__
   (November 4-8th, 2019)

.. |Seibert Heirarchy of Needs SciPy 2019| image:: https://pbs.twimg.com/media/EBYojw8XUAERJhZ?format=png

.. |check| raw:: html

    <input checked=""  type="checkbox" disabled="true">

.. |uncheck| raw:: html

    <input type="checkbox" disabled="true">
