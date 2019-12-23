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
Like the rest of the Python community, as of January 2020 ``pyhf`` no longer supports Python 2.
The last release of ``pyhf`` that was compatible with Python 2.7 is `v0.3.2 <https://pypi.org/project/pyhf/0.3.2/>`__.


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
