Developing
==========

To develop, we suggest using `virtual environments <https://virtualenvwrapper.readthedocs.io/en/latest/>`__ together with ``pip`` or using `pipenv <https://pipenv.readthedocs.io/en/latest/>`__. To get all necessary packages for development::

    pip install --ignore-installed -U -e .[complete]

Then setup the Git pre-commit hook for `Black <https://github.com/ambv/black>`__  by running::

    pre-commit install
