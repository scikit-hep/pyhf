FAQ
===

Frequently Asked Questions about :code:`pyhf` and its use.

Questions
---------

Where can I ask questions about ``pyhf`` use?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you have a question about the use of ``pyhf`` not covered in the `documentation <https://scikit-hep.org/pyhf/>`__, please ask a question on `Stack Overflow <https://stackoverflow.com/questions/tagged/pyhf>`__ with the ``[pyhf]`` tag, which the ``pyhf`` dev team `watches <https://stackoverflow.com/questions/tagged/pyhf?sort=Newest&filters=NoAcceptedAnswer&edited=true>`__.

.. raw:: html

  <p align="center">
  <a href="https://stackoverflow.com/questions/tagged/pyhf">
  <img src="https://cdn.sstatic.net/Sites/stackoverflow/company/img/logos/so/so-logo.png" alt="Stack Overflow pyhf tag" width="50%"/>
  </a>
  </p>

If you believe you have found a bug in ``pyhf``, please report it in the `GitHub Issues <https://github.com/scikit-hep/pyhf/issues/new?template=Bug-Report.md&labels=bug&title=Bug+Report+:+Title+Here>`__.

Is it possible to set the backend from the CLI?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes.
Use the :code:`--backend` option for :code:`pyhf cls` to specify a tensor backend.
The default backend is NumPy.
For more information see :code:`pyhf cls --help`.

Does ``pyhf`` support Python 2?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
No.
Like the rest of the Python community, as of January 2020 the latest releases of ``pyhf`` no longer support Python 2.
The last release of ``pyhf`` that was compatible with Python 2.7 is `v0.3.4 <https://pypi.org/project/pyhf/0.3.4/>`__, which can be installed with

    .. code-block:: console

        python -m pip install pyhf~=0.3

I only have access to Python 2. How can I use ``pyhf``?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is recommended that ``pyhf`` is used as a standalone step in any analysis, and its environment need not be the same as the rest of the analysis.
As Python 2 is not supported it is suggested that you setup a Python 3 runtime on whatever machine you're using.
If you're using a cluster, talk with your system administrators to get their help in doing so.
If you are unable to get a Python 3 runtime, versioned Docker images of ``pyhf`` are distributed through `Docker Hub <https://hub.docker.com/r/pyhf/pyhf>`__.

Once you have Python 3 installed, see the :ref:`installation` page to get started.

How is ``pyhf`` typeset?
~~~~~~~~~~~~~~~~~~~~~~~~

As you may have guessed from this page, ``pyhf`` is typeset in all lowercase.
This is largely historical, as the core developers had just always typed it that way and it seemed a bit too short of a library name to write as ``PyHF``.
When typesetting in LaTeX the developers recommend introducing the command

    .. code-block:: latex

        \newcommand{\pyhf}{\texttt{pyhf}}

If the journal you are publishing in requires you to use ``textsc`` for software names it is okay to instead use

    .. code-block:: latex

        \newcommand{\pyhf}{\textsc{pyhf}}

Troubleshooting
---------------

- :code:`import torch` or :code:`import pyhf` causes a :code:`Segmentation fault (core dumped)`

    This is may be the result of a conflict with the NVIDIA drivers that you
    have installed on your machine.  Try uninstalling and completely removing
    all of them from your machine

    .. code-block:: console

        # On Ubuntu/Debian
        sudo apt-get purge nvidia*

    and then installing the latest versions.
