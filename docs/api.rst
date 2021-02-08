Python API
==========

Top-Level
---------

.. currentmodule:: pyhf

.. autosummary::
   :toctree: _generated/

   default_backend
   default_optimizer
   tensorlib
   optimizer
   get_backend
   set_backend
   readxml
   writexml

Probability Distribution Functions (PDFs)
-----------------------------------------

.. currentmodule:: pyhf.probability

.. autosummary::
   :toctree: _generated/
   :nosignatures:
   :template: modifierclass.rst

   Normal
   Poisson
   Independent
   Simultaneous

Making Models from PDFs
-----------------------

.. currentmodule:: pyhf

.. autosummary::
   :toctree: _generated/
   :nosignatures:
   :template: modifierclass.rst

   ~pdf.Model
   ~pdf._ModelConfig
   ~workspace.Workspace
   ~patchset.PatchSet
   ~patchset.Patch
   simplemodels.hepdata_like

Backends
--------

The computational backends that :code:`pyhf` provides interfacing for the vector-based calculations.

.. currentmodule:: pyhf.tensor

.. autosummary::
   :toctree: _generated/
   :nosignatures:
   :template: modifierclass.rst

   numpy_backend.numpy_backend
   pytorch_backend.pytorch_backend
   tensorflow_backend.tensorflow_backend
   jax_backend.jax_backend

Optimizers
----------

.. currentmodule:: pyhf.optimize

.. autosummary::
   :toctree: _generated/
   :nosignatures:
   :template: modifierclass.rst

   mixins.OptimizerMixin
   opt_scipy.scipy_optimizer
   opt_minuit.minuit_optimizer

Modifiers
---------

.. currentmodule:: pyhf.modifiers

.. autosummary::
   :toctree: _generated/
   :nosignatures:
   :template: modifierclass.rst

   histosys
   normfactor
   normsys
   shapefactor
   shapesys
   staterror

Interpolators
-------------

.. currentmodule:: pyhf.interpolators

.. autosummary::
   :toctree: _generated/
   :nosignatures:
   :template: modifierclass.rst

   code0
   code1
   code2
   code4
   code4p

Inference
---------

.. currentmodule:: pyhf.infer

.. autosummary::
   :toctree: _generated/
   :template: modifierclass.rst

   test_statistics.q0
   test_statistics.qmu
   test_statistics.qmu_tilde
   test_statistics.tmu
   test_statistics.tmu_tilde
   mle.nll
   mle.fit
   mle.fixed_poi_fit
   hypotest
   intervals.upperlimit
   calculators.generate_asimov_data
   calculators.AsymptoticTestStatDistribution
   calculators.EmpiricalDistribution
   calculators.AsymptoticCalculator
   calculators.ToyCalculator
   utils.create_calculator
   utils.get_test_stat

Exceptions
----------

Various exceptions, apart from standard python exceptions, that are raised from using the :code:`pyhf` API.

.. currentmodule:: pyhf.exceptions

.. autosummary::
   :toctree: _generated/
   :nosignatures:
   :template: modifierclass.rst

   InvalidMeasurement
   InvalidNameReuse
   InvalidSpecification
   InvalidPatchSet
   InvalidPatchLookup
   PatchSetVerificationError
   InvalidWorkspaceOperation
   InvalidModel
   InvalidModifier
   InvalidInterpCode
   ImportBackendError
   InvalidBackend
   InvalidOptimizer
   InvalidPdfParameters
   InvalidPdfData

Utilities
---------

.. currentmodule:: pyhf.utils

.. autosummary::
   :toctree: _generated/

   load_schema
   validate
   options_from_eqdelimstring
   digest

Contrib
-------

.. currentmodule:: pyhf.contrib

.. autosummary::
   :toctree: _generated/

   viz.brazil
   utils.download
