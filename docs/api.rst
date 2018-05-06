API
===

Top-Level
---------

.. currentmodule:: pyhf

.. autosummary::
   :toctree: _generated/
   :nosignatures:

   default_backend
   default_optimizer
   generate_asimov_data
   get_backend
   hfpdf
   loglambdav
   modelconfig
   pvals_from_teststat
   qmu
   runOnePoint
   set_backend
   tensorlib

Backends
--------

.. currentmodule:: pyhf.tensor

.. autosummary::
   :toctree: _generated/
   :nosignatures:
   :template: modifierclass.rst

   mxnet_backend
   numpy_backend
   pytorch_backend
   tensorflow_backend

Optimizers
----------

.. currentmodule:: pyhf.optimize

.. autosummary::
   :toctree: _generated/
   :nosignatures:
   :template: modifierclass.rst

   pytorch_optimizer
   scipy_optimizer
   tflow_optimizer

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

Exceptions
----------

.. currentmodule:: pyhf.exceptions

.. autosummary::
   :toctree: _generated/
   :nosignatures:
   :template: modifierclass.rst

   InvalidInterpCode
   InvalidModifier

