=============
Release Notes
=============

|release v0.5.4|_
=================

This is a patch release from v0.5.3 → v0.5.4.

Fixes
-----

* Require ``uproot3`` instead of ``uproot`` ``v3.X`` releases to avoid conflicts when
  ``uproot4`` is installed in an environment with ``uproot`` ``v3.X`` installed and
  namespace conflicts with ``uproot-methods``.
  Adoption of ``uproot3`` in ``v0.5.4`` will ensure ``v0.5.4`` works far into the future
  if XML and ROOT I/O through uproot is required.

**Example:**

Without the ``v0.5.4`` patch release there is a regression in using ``uproot`` ``v3.X``
and ``uproot4`` in the same environment (which was swiftly identified and patched by the
fantastic ``uproot`` team)

.. code-block:: shell

   $ python -m pip install "pyhf[xmlio]<0.5.4"
   $ python -m pip list | grep "pyhf\|uproot"
   pyhf           0.5.3
   uproot         3.13.1
   uproot-methods 0.8.0
   $ python -m pip install uproot4
   $ python -m pip list | grep "pyhf\|uproot"
   pyhf           0.5.3
   uproot         4.0.0
   uproot-methods 0.8.0
   uproot4        4.0.0

this is resolved in ``v0.5.4`` with the requirement of ``uproot3``

.. code-block:: shell

   $ python -m pip install "pyhf[xmlio]>=0.5.4"
   $ python -m pip list | grep "pyhf\|uproot"
   pyhf            0.5.4
   uproot3         3.14.1
   uproot3-methods 0.10.0
   $ python -m pip install uproot4 # or uproot
   $ python -m pip list | grep "pyhf\|uproot"
   pyhf            0.5.4
   uproot          4.0.0
   uproot3         3.14.1
   uproot3-methods 0.10.0
   uproot4         4.0.0

|release v0.5.3|_
=================

This is a patch release from v0.5.2 → v0.5.3.

Fixes
-----

* Workspaces are now immutable
* ShapeFactor support added to XML reading and writing
* An error is raised if a fit initialization parameter is outside of its bounds
  (preventing hypotest with POI outside of bounds)

Features
--------

Python API
~~~~~~~~~~

* Inverting hypothesis tests to get upper limits now has an API with
  ``pyhf.infer.intervals.upperlimit``
* Building workspaces from a model and data added with ``pyhf.workspace.build``

CLI API
~~~~~~~

* Added CLI API for ``pyhf.infer.fit``: ``pyhf fit``
* pyhf combine now allows for merging channels: ``pyhf combine --merge-channels --join <join option>``
* Added utility to download archived pyhf pallets (workspaces + patchsets) to contrib module: ``pyhf contrib download``

.. |release v0.5.4| replace:: ``v0.5.4``
.. _`release v0.5.4`: https://github.com/scikit-hep/pyhf/releases/tag/v0.5.4

.. |release v0.5.3| replace:: ``v0.5.3``
.. _`release v0.5.3`: https://github.com/scikit-hep/pyhf/releases/tag/v0.5.3
