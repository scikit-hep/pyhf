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

   opt_pytorch.pytorch_optimizer
   opt_scipy.scipy_optimizer
   opt_tflow.tflow_optimizer
   opt_jax.jax_optimizer
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

   hypotest
   test_statistics.qmu
   mle.twice_nll
   mle.fit
   mle.fixed_poi_fit
   calculators.generate_asimov_data
   calculators.AsymptoticTestStatDistribution
   calculators.AsymptoticCalculator

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
   InvalidWorkspaceOperation
   InvalidModel
   InvalidModifier
   InvalidInterpCode
   ImportBackendError
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
