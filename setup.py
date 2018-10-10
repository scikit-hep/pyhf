#!/usr/bin/env python

from setuptools import setup, find_packages
from pyhf.version import __version__

extras_require = {
    'tensorflow': [
        'tensorflow>=1.10.0',
        'tensorflow-probability==0.3.0',
        'numpy<=1.14.5,>=1.14.0',  # Lower of 1.14.0 instead of 1.13.3 to ensure doctest pass
        'setuptools<=39.1.0',
    ],
    'torch': [
        'torch>=0.4.0'
    ],
    'mxnet': [
        'mxnet>=1.0.0',
        'requests<2.19.0,>=2.18.4',
        'numpy<1.15.0,>=1.8.2',
        'requests<2.19.0,>=2.18.4',
    ],
    # 'dask': [
    #     'dask[array]'
    # ],
    'xmlimport': [
        'uproot',
    ],
    'minuit': [
        'iminuit'
    ],
    'develop': [
        'pyflakes',
        'pytest>=3.5.1',
        'pytest-cov>=2.5.1',
        'pytest-benchmark[histogram]',
        'pytest-console-scripts',
        'python-coveralls',
        'coverage>=4.0',  # coveralls
        'matplotlib',
        'jupyter',
        'nbdime',
        'uproot>=3.0.0',
        'papermill',
        'graphviz',
        'sphinx',
        'sphinxcontrib-bibtex',
        'sphinxcontrib-napoleon',
        'sphinx_rtd_theme',
        'nbsphinx',
        'm2r',
        'jsonpatch',
        'ipython<7',  # jupyter_console and ipython clash in dependency requirement -- downgrade ipython for now
    ]
}
extras_require['complete'] = sorted(set(sum(extras_require.values(), [])))

setup(
    name='pyhf',
    version=__version__,
    description='(partial) pure python histfactory implementation',
    url='https://github.com/diana-hep/pyhf',
    author='Lukas Heinrich',
    author_email='lukas.heinrich@cern.ch',
    license='Apache',
    keywords='physics fitting numpy scipy tensorflow pytorch mxnet dask',
    classifiers=[
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
    ],
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*, !=3.5.*",
    install_requires=[
        'scipy',  # requires numpy, which is required by pyhf, tensorflow, and mxnet
        'click>=6.0',  # for console scripts,
        'tqdm',  # for readxml
        'six',  # for modifiers
        'jsonschema>=v3.0.0a2',  # for utils, alpha-release for draft 6
        'jsonpatch'
    ],
    extras_require=extras_require,
    entry_points={
        'console_scripts': ['pyhf=pyhf.commandline:pyhf']
    },
    dependency_links=[]
)
