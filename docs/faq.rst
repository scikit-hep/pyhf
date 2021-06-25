.. _sec:faq:

FAQ
===

Frequently Asked Questions about :code:`pyhf` and its use.

Questions
---------

Where can I ask questions about ``pyhf`` use?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you have a question about the use of ``pyhf`` not covered in the `documentation <https://pyhf.readthedocs.io/>`__, please ask a question on the `GitHub Discussions <https://github.com/scikit-hep/pyhf/discussions>`__.

If you believe you have found a bug in ``pyhf``, please report it in the `GitHub Issues <https://github.com/scikit-hep/pyhf/issues/new?template=Bug-Report.md&labels=bug&title=Bug+Report+:+Title+Here>`__.

How can I get updates on ``pyhf``?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you're interested in getting updates from the ``pyhf`` dev team and release
announcements you can join the |pyhf-announcements mailing list|_.

.. |pyhf-announcements mailing list| replace:: ``pyhf-announcements`` mailing list
.. _pyhf-announcements mailing list: https://groups.google.com/group/pyhf-announcements/subscribe

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

I validated my workspace by comparing ``pyhf`` and ``HistFactory``, and while the expected CLs matches, the observed CLs is different. Why is this?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Make sure you're using the right test statistic (:math:`q` or :math:`\tilde{q}`) in both situations.
In ``HistFactory``, the asymptotics calculator, for example, will do something more involved for the observed CLs if you choose a different test statistic.

I ran validation to compare ``HistFitter`` and ``pyhf``, but they don't match exactly. Why not?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``pyhf`` is validated against ``HistFactory``.
``HistFitter`` makes some particular implementation choices that ``pyhf`` doesn't reproduce.
Instead of trying to compare ``pyhf`` and ``HistFitter`` you should try to validate them both against ``HistFactory``.

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

Why use Python?
~~~~~~~~~~~~~~~

As of the late 2010's Python is widely considered the lingua franca of machine learning
libraries, and is sufficiently high-level and expressive for physicists of various computational
skill backgrounds to use.
Using Python as the language for development allows for the distribution of the software
--- as both source files and binary distributions --- through the Python Package Index (PyPI)
and Conda-forge, which significantly lowers the barrier for use as compared to ``C++``.
Additionally, a 2017 `DIANA/HEP <https://diana-hep.org/>`_ study :cite:`faq-feickert-diana-fellowship-report`
demonstrated the graph structure and automatic differentiation abilities of machine learning
frameworks allowed them to be quite effective tools for statistical fits.
As the frameworks considered in this study (TensorFlow, PyTorch, MXNet) all provided
low-level Python APIs to the libraries this made Python an obvious choice for a common
high-level control language.
Given all these considerations, Python was chosen as the development language.

How did ``pyhf`` get started?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In 2017 Lukas Heinrich was discussing with colleauge Holger Schulz how it would be convienent
to share and produce statistical results from LHC experiements if they were able to be
created with tools that didn't require the large ``C++`` dependencies and tooling expertise as
:math:`\HiFa{}`.
Around the same time that Lukas began thinking on these ideas, Matthew Feickert was working on
`a DIANA/HEP fellowship <https://twitter.com/SMUPhysics/status/861584474638766080>`_ with
Kyle Cranmer (co-author of :math:`\HiFa{}`) to study if the graph structure and automatic
differentiation abilities of machine learning frameworks would allow them to be effective
tools for statistical fits.
Lukas would give helpful friendly advice on Matthew's project and one night [1]_ over dinner
in CERN's R1 cafeteria the two were discussing the idea of implimenting :math:`\HiFa{}`
in Python using machine learning libraries to drive the computation.
Continuing the discussion in Lukas's office, Lukas showed Matthew that the core statistical
machinery could be implimented rather succinctly, and that night
`proceeded to do so <https://github.com/scikit-hep/pyhf/commit/fd32503fb760f070a4047cb867757458b1687599>`_
and |dubbed the project pyhf|_.

Matthew joined him on the project to begin development and by April 2018 Giordon Stark had
learned about the project and began making contributions, quickly becoming
`the third core developer <https://twitter.com/KyleCranmer/status/1052186117452259328>`_.
The first physics paper to use ``pyhf`` followed closely in October 2018
:cite:`faq-Heinrich:2018nip`, making Lukas and Holger's original conversations a reality.
``pyhf`` was founded on the ideas of open contributions and community software and continues
in that mission today as a `Scikit-HEP project <https://scikit-hep.org/>`_, with an open
invitation for community contributions and new developers.

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

Footnotes
~~~~~~~~~

.. [1]
   24 January, 2018

Bibliography
~~~~~~~~~~~~

.. bibliography:: bib/docs.bib bib/use_citations.bib
   :filter: docname in docnames
   :style: plain
   :keyprefix: faq-
   :labelprefix: faq-

.. |dubbed the project pyhf| replace:: dubbed the project ``pyhf``
.. _`dubbed the project pyhf`: https://twitter.com/lukasheinrich_/status/956809112674885632
