FAQ
===

Frequently Asked Questions about :code:`pyhf` and its use.

Questions
---------

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
