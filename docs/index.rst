.. pyhf documentation master file, created by
   sphinx-quickstart on Fri Feb  9 11:58:49 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::
   :hidden:

   intro
   likelihood
   learn
   examples
   outreach
   installation
   development
   faq
   babel
   cli
   api
   citations
   governance/ROADMAP
   release-notes
   contributors

.. raw:: html

   <a class="github-fork-ribbon right-top fixed" href="https://github.com/scikit-hep/pyhf/" data-ribbon="View me on GitHub" title="View me on GitHub">View me on GitHub</a>


.. raw:: html

   <p id="dev-version"><strong>Warning:</strong> This is a development version. The latest stable version is at <a href="https://pyhf.readthedocs.io/">ReadTheDocs</a>.</p>

..
  Comment: Splice the JupyterLite example into the README by looking for a particular comment

.. include:: ../README.rst
    :end-before: Comment: JupyterLite segment goes here in docs

.. include:: jupyterlite/jupyterlite.rst

.. include:: ../README.rst
    :start-after: Comment: JupyterLite segment goes here in docs
    :end-before: Comment: JupyterLite Hello World start

.. include:: jupyterlite/hello_world.rst

.. include:: ../README.rst
    :start-after: Comment: JupyterLite Hello World end
    :end-before: Comment: JupyterLite Hello World JSON start

.. include:: jupyterlite/hello_world_json.rst

.. include:: ../README.rst
    :start-after: Comment: JupyterLite Hello World JSON end
    :end-before: Comment: JupyterLite one bin example start

.. include:: jupyterlite/one_bin.rst

.. include:: ../README.rst
    :start-after: Comment: JupyterLite one bin example end

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
