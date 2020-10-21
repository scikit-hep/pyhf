.. _sec:likelihood:

Likelihood Specification
========================

The structure of the JSON specification of models follows closely the
original XML-based specification :cite:`likelihood-Cranmer:1456844`.

Workspace
---------

.. literalinclude:: ../src/pyhf/schemas/1.0.0/workspace.json
   :language: json

The overall document in the above code snippet describes a *workspace*, which includes

* **channels**: The channels in the model, which include a description of the samples
  within each channel and their possible parametrised modifiers.
* **measurements**: A set of measurements, which define among others the parameters of
  interest for a given statistical analysis objective.
* **observations**: The observed data, with which a likelihood can be constructed from the model.

A workspace consists of the channels, one set of observed data, but can
include multiple measurements. If provided a JSON file, one can quickly
check that it conforms to the provided workspace specification as follows:

.. code:: python

   import json, requests, jsonschema
   workspace = json.load(open('/path/to/analysis_workspace.json'))
   # if no exception is raised, it found and parsed the schema
   schema = requests.get('https://scikit-hep.org/pyhf/schemas/1.0.0/workspace.json').json()
   # If no exception is raised by validate(), the instance is valid.
   jsonschema.validate(instance=workspace, schema=schema)


.. _ssec:channel:

Channel
-------

A channel is defined by a channel name and a list of samples :cite:`likelihood-schema_defs`.

.. code:: json

    {
        "channel": {
            "type": "object",
            "properties": {
                "name": { "type": "string" },
                "samples": { "type": "array", "items": {"$ref": "#/definitions/sample"}, "minItems": 1 }
            },
            "required": ["name", "samples"],
            "additionalProperties": false
        },
    }

The Channel specification consists of a list of channel descriptions.
Each channel, an analysis region encompassing one or more measurement
bins, consists of a ``name`` field and a ``samples`` field (see :ref:`ssec:channel`), which
holds a list of sample definitions (see :ref:`ssec:sample`). Each sample definition in
turn has a ``name`` field, a ``data`` field for the nominal event rates
for all bins in the channel, and a ``modifiers`` field of the list of
modifiers for the sample.

.. _ssec:sample:

Sample
------

A sample is defined by a sample name, the sample event rate, and a list of modifiers :cite:`likelihood-schema_defs`.

.. _lst:schema:sample:

.. code:: json

    {
        "sample": {
            "type": "object",
            "properties": {
                "name": { "type": "string" },
                "data": { "type": "array", "items": {"type": "number"}, "minItems": 1 },
                "modifiers": {
                    "type": "array",
                    "items": {
                        "anyOf": [
                            { "$ref": "#/definitions/modifier/histosys" },
                            { "$ref": "#/definitions/modifier/lumi" },
                            { "$ref": "#/definitions/modifier/normfactor" },
                            { "$ref": "#/definitions/modifier/normsys" },
                            { "$ref": "#/definitions/modifier/shapefactor" },
                            { "$ref": "#/definitions/modifier/shapesys" },
                            { "$ref": "#/definitions/modifier/staterror" }
                        ]
                    }
                }
            },
            "required": ["name", "data", "modifiers"],
            "additionalProperties": false
        },
    }

Modifiers
---------

The modifiers that are applicable for a given sample are encoded as a
list of JSON objects with three fields. A name field, a type field
denoting the class of the modifier, and a data field which provides the
necessary input data as denoted in :ref:`tab:modifiers_and_constraints`.

Based on the declared modifiers, the set of parameters and their
constraint terms are derived implicitly as each type of modifier
unambiguously defines the constraint terms it requires. Correlated shape
modifiers and normalisation uncertainties have compatible constraint
terms and thus modifiers can be declared that *share* parameters by
re-using a name [1]_ for multiple modifiers. That is, a variation of a
single parameter causes a shift within sample rates due to both shape
and normalisation variations.

We review the structure of each modifier type below.

Uncorrelated Shape (shapesys)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To construct the constraint term, the relative uncertainties
:math:`\sigma_b` are necessary for each bin. Therefore, we record the
absolute uncertainty as an array of floats, which combined with the
nominal sample data yield the desired :math:`\sigma_b`. An example is
shown below:

.. code:: json

   { "name": "mod_name", "type": "shapesys", "data": [1.0, 1.5, 2.0] }

An example of an uncorrelated shape modifier with three absolute uncertainty
terms for a 3-bin channel.

.. warning::

   Nuisance parameters will not be allocated for any bins where either

     * the samples nominal expected rate is zero, or
     * the absolute uncertainty is zero.

   These values are, in the context of uncorrelated shape uncertainties,
   unphysical. If this situation occurs, one needs to go back and understand
   the inputs as this is undefined behavior in HistFactory.

The previous example will allocate three nuisance parameters for ``mod_name``.
The following example will allocate only two nuisance parameters for a 3-bin
channel:

.. code:: json

   { "name": "mod_name", "type": "shapesys", "data": [1.0, 0.0, 2.0] }

Correlated Shape (histosys)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

This modifier represents the same source of uncertainty which has a
different effect on the various sample shapes, hence a correlated shape.
To implement an interpolation between sample distribution shapes, the
distributions with a "downward variation" ("lo") associated with
:math:`\alpha=-1` and an "upward variation" ("hi") associated with
:math:`\alpha=+1` are provided as arrays of floats. An example is shown
below:

.. code:: json

   { "name": "mod_name", "type": "histosys", "data": {"hi_data": [20,15], "lo_data": [10, 10]} }

An example of a correlated shape modifier with absolute shape variations for a 2-bin channel.

Normalisation Uncertainty (normsys)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The normalisation uncertainty modifies the sample rate by a overall
factor :math:`\kappa(\alpha)` constructed as the interpolation between
downward ("lo") and upward ("hi") as well as the nominal setting, i.e.
:math:`\kappa(-1) = \kappa_{\alpha=-1}`, :math:`\kappa(0) = 1` and
:math:`\kappa(+1) = \kappa_{\alpha=+1}`. In the modifier definition we record
:math:`\kappa_{\alpha=+1}` and :math:`\kappa_{\alpha=-1}` as floats. An
example is shown below:

.. code:: json

   { "name": "mod_name", "type": "normsys", "data": {"hi": 1.1, "lo": 0.9} }

An example of a normalisation uncertainty modifier with scale factors recorded for the up/down variations of an :math:`n`-bin channel.

MC Statistical Uncertainty (staterror)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As the sample counts are often derived from Monte Carlo (MC) datasets, they
necessarily carry an uncertainty due to the finite sample size of the datasets.
As explained in detail in :cite:`likelihood-Cranmer:1456844`, adding uncertainties for
each sample would yield a very large number of nuisance parameters with limited
utility. Therefore a set of bin-wise scale factors :math:`\gamma_b` is
introduced to model the overall uncertainty in the bin due to MC statistics.
The constrained term is constructed as a set of Gaussian constraints with a
central value equal to unity for each bin in the channel. The scales
:math:`\sigma_b` of the constraint are computed from the individual
uncertainties of samples defined within the channel relative to the total event
rate of all samples: :math:`\delta_{csb} = \sigma_{csb}/\sum_s \nu^0_{scb}`. As
not all samples are within a channel are estimated from MC simulations, only
the samples with a declared statistical uncertainty modifier enter the sum. An
example is shown below:

.. code:: json

   { "name": "mod_name", "type": "staterror", "data": [0.1] }

An example of a statistical uncertainty modifier.

Luminosity (lumi)
~~~~~~~~~~~~~~~~~

Sample rates derived from theory calculations, as opposed to data-driven
estimates, are scaled to the integrated luminosity corresponding to the
observed data. As the luminosity measurement is itself subject to an
uncertainty, it must be reflected in the rate estimates of such samples.  As
this modifier is of global nature, no additional per-sample information is
required and thus the data field is nulled. This uncertainty is relevant, in
particular, when the parameter of interest is a signal cross-section. The
luminosity uncertainty :math:`\sigma_\lambda` is provided as part of the
parameter configuration included in the measurement specification discussed
in :ref:`ssec:measurements`.  An example is shown below:

.. code:: json

   { "name": "mod_name", "type": "lumi", "data": null }

An example of a luminosity modifier.

Unconstrained Normalisation (normfactor)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The unconstrained normalisation modifier scales the event rates of a
sample by a free parameter :math:`\mu`. Common use cases are the signal
rate of a possible BSM signal or simultaneous in-situ measurements of
background samples. Such parameters are frequently the parameters of
interest of a given measurement. No additional per-sample data is
required. An example is shown below:

.. code:: json

   { "name": "mod_name", "type": "normfactor", "data": null }

An example of a normalisation modifier.

Data-driven Shape (shapefactor)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In order to support data-driven estimation of sample rates (e.g. for
multijet backgrounds), the data-driven shape modifier adds free,
bin-wise multiplicative parameters. Similarly to the normalisation
factors, no additional data is required as no constraint is defined. An
example is shown below:

.. code:: json

   { "name": "mod_name", "type": "shapefactor", "data": null }

An example of an uncorrelated shape modifier.

Data
----

The data provided by the analysis are the observed data for each channel
(or region). This data is provided as a mapping from channel name to an
array of floats, which provide the observed rates in each bin of the
channel. The auxiliary data is not included as it is an input to the
likelihood that does not need to be archived and can be determined
automatically from the specification. An example is shown below:

.. _lst:example:data:

.. code:: json

   { "chan_name_one": [10, 20], "chan_name_two": [4, 0]}

An example of channel data.

.. _ssec:measurements:

Measurements
------------

Given the data and the model definitions, a measurement can be defined.
In the current schema, the measurements defines the name of the
parameter of interest as well as parameter set configurations.  [2]_
Here, the remaining information not covered through the channel
definition is provided, e.g. for the luminosity parameter. For all
modifiers, the default settings can be overridden where possible:

* **inits**: Initial value of the parameter.
* **bounds**: Interval bounds of the parameter.
* **auxdata**: Auxiliary data for the associated constraint term.
* **sigmas**: Associated uncertainty of the parameter.

An example is shown below:

.. code:: json

   {
       "name": "MyMeasurement",
       "config": {
           "poi": "SignalCrossSection", "parameters": [
               { "name":"lumi", "auxdata":[1.0],"sigmas":[0.017], "bounds":[[0.915,1.085]],"inits":[1.0] },
               { "name":"mu_ttbar", "bounds":[[0, 5]] },
               { "name":"rw_1CR", "fixed":true }
           ]
       }
   }

An example of a measurement. This measurement, which scans over the parameter of interest ``SignalCrossSection``, is setting configurations for the luminosity modifier, changing the default bounds for the normfactor modifier named ``mu_ttbar``, and specifying that the modifier ``rw_1CR`` is held constant (``fixed``).

.. _ssec:observations:

Observations
------------

This is what we evaluate the hypothesis testing against, to determine the
compatibility of signal+background hypothesis to the background-only
hypothesis. This is specified as a list of objects, with each object structured
as

* **name**: the channel for which the observations are recorded
* **data**: the bin-by-bin observations for the named channel

An example is shown below:

.. code:: json

   {
       "name": "channel1",
       "data": [110.0, 120.0]
   }

An example of an observation. This observation recorded for a 2-bin channel ``channel1``, has values ``110.0`` and ``120.0``.

Toy Example
-----------

.. # N.B. If the following literalinclude is changed test_examples.py must be changed accordingly
.. literalinclude:: ./examples/json/2-bin_1-channel.json
   :language: json

In the above example, we demonstrate a simple measurement of a
single two-bin channel with two samples: a signal sample and a background
sample. The signal sample has an unconstrained normalisation factor
:math:`\mu`, while the background sample carries an uncorrelated shape
systematic controlled by parameters :math:`\gamma_1` and :math:`\gamma_2`. The
background uncertainty for the bins is 10% and 20% respectively.

Additional Material
-------------------

Footnotes
~~~~~~~~~

.. [1]
   The name of a modifier specifies the parameter set it is controlled
   by. Modifiers with the same name share parameter sets.

.. [2]
   In this context a parameter set corresponds to a named
   lower-dimensional subspace of the full parameters :math:`\fullset`.
   In many cases these are one-dimensional subspaces, e.g. a specific
   interpolation parameter :math:`\alpha` or the luminosity parameter
   :math:`\lambda`. For multi-bin channels, however, e.g. all bin-wise
   nuisance parameters of the uncorrelated shape modifiers are grouped
   under a single name. Therefore in general a parameter set definition
   provides arrays of initial values, bounds, etc.

Bibliography
~~~~~~~~~~~~

.. bibliography:: bib/docs.bib
   :filter: docname in docnames
   :style: plain
   :keyprefix: likelihood-
   :labelprefix: likelihood-
