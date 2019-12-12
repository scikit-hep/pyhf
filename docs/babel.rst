Translations
============
One key goal of ``pyhf`` is to provide seamless translations between other statistical frameworks and ``pyhf``.
This page details the various ways to translate from a tool you might already be using as part of an existing analysis to ``pyhf``.
Many of these solutions involve extracting out the ``HistFactory`` workspace and then running `pyhf xml2json <cli.html#pyhf-xml2json>`_ which provides a single JSON workspace that can be loaded directly into ``pyhf``.

HistFitter
----------

In order to go from ``HistFitter`` to ``pyhf``, the first step is to extract out the ``HistFactory`` workspaces. Assuming you have an existing configuration file, ``config.py``, you likely run an exclusion fit like so:

.. code:: bash

  HistFitter.py -f -D "before,after,corrMatrix" -F excl config.py

The name of output workspace files depends on four parameters you define in your ``config.py``:

- ``analysisName`` is from ``configMgr.analysisName``
- ``prefix`` is defined in ``configMgr.addFitConfig({prefix})``
- ``measurementName`` is the first measurement you define via ``fitConfig.addMeasurement(name={measurementName},...)``
- ``channelName`` are the names of channels you define via ``fitConfig.addChannel("cuts", [{channelName}], ...)``
- ``cachePath`` is where ``HistFitter`` stores the cached histograms, defined by ``configMgr.histCacheFile`` which defaults to ``data/{analysisName}.root``

To dump the HistFactory workspace, you will modify the above to skip the fit ``-f`` and plotting ``-D`` so you end up with

.. code:: bash

  HistFitter.py -wx -F excl config.py

The ``-w`` flag tells ``HistFitter`` to (re)create the ``HistFactory`` workspace stored in ``results/{analysisName}/{prefix}_combined_{measurementName}.root``.
The ``-x`` flag tells ``HistFitter`` to dump the XML files into ``config/{analysisName}/``, with the top-level file being ``{prefix}.xml`` and all other files being ``{prefix}_{channelName}_cuts.xml``.

Typically, ``prefix = 'FitConfig'`` and ``measurementName = 'NormalMeasurement'``. For example, if the following exists in your ``config.py``

.. code:: python

  from configManager import configMgr
  # ...
  configMgr.analysisName = '3b_tag21.2.27-1_RW_ExpSyst_36100_multibin_bkg'
  configMgr.histCacheFile = 'cache/{0:s}.root'.format(configMgr.analysisName)
  # ...
  fitConfig = configMgr.addFitConfig("Excl")
  # ...
  channel = fitConfig.addChannel("cuts", ['SR_0L'], 1, 0.5, 1.5)
  # ...
  meas1=fitConfig.addMeasurement(name="DefaultMeasurement",lumi=1.0,lumiErr=0.029)
  meas1.addPOI("mu_SIG1")
  # ...
  meas2=fitConfig.addMeasurement(name="DefaultMeasurement",lumi=1.0,lumiErr=0.029)
  meas2.addPOI("mu_SIG2")

Then, you expect the following files to be made:

- ``config/3b_tag21.2.27-1_RW_ExpSyst_36100_multibin_bkg/Excl.xml``
- ``config/3b_tag21.2.27-1_RW_ExpSyst_36100_multibin_bkg/Excl_SR_0L_cuts.xml``
- ``cache/3b_tag21.2.27-1_RW_ExpSyst_36100_multibin_bkg.root``
- ``results/3b_tag21.2.27-1_RW_ExpSyst_36100_multibin_bkg/Excl_combined_DefaultMeasurement.root``

These are all the files you need in order to use `pyhf xml2json <cli.html#pyhf-xml2json>`_. At this point, you could run

.. code:: bash

    pyhf xml2json config/3b_tag21.2.27-1_RW_ExpSyst_36100_multibin_bkg/Excl.xml

which will read all of the XML files and load the histogram data from the histogram cache.

The ``HistFactory`` workspace in ``results/`` contains all of the information necessary to rebuild the XML files again. For debugging purposes, the ``pyhf`` developers will often ask for your workspace file, which means ``results/3b_tag21.2.27-1_RW_ExpSyst_36100_multibin_bkg/Excl_combined_DefaultMeasurement.root``. If you want to generate the XML, you can open this file in ``ROOT`` and run ``DefaultMeasurement->PrintXML()`` which puts all of the XML files into the current directory you are in.
