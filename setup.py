#!/usr/bin/env python

from setuptools import setup, find_packages
from os import path
import sys

this_directory = path.abspath(path.dirname(__file__))
if sys.version_info.major < 3:
    from io import open
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as readme_md:
    long_description = readme_md.read()

extras_require = {
    'tensorflow': [
        'tensorflow>=1.12.0',
        'tensorflow-probability>=0.5.0',
        'numpy<=1.14.5,>=1.14.0',  # Lower of 1.14.0 instead of 1.13.3 to ensure doctest pass
        'setuptools<=39.1.0',
    ],
    'torch': ['torch>=1.0.0'],
    'mxnet': [
        'mxnet>=1.0.0',
        'requests<2.19.0,>=2.18.4',
        'numpy<1.15.0,>=1.8.2',
        'requests<2.19.0,>=2.18.4',
    ],
    # 'dask': [
    #     'dask[array]'
    # ],
    'xmlimport': ['uproot'],
    'minuit': ['iminuit'],
    'develop': [
        'pyflakes',
        'pytest<4.0.0,>=3.5.1',
        'pytest-cov>=2.5.1',
        'pytest-mock',
        'pytest-benchmark[histogram]',
        'pytest-console-scripts',
        'python-coveralls',
        'coverage>=4.0',  # coveralls
        'matplotlib',
        'jupyter',
        'nbdime',
        'uproot>=3.3.0',
        'papermill>=0.16.0',
        'graphviz',
        'bumpversion',
        'sphinx',
        'sphinxcontrib-bibtex',
        'sphinxcontrib-napoleon',
        'sphinx_rtd_theme',
        'nbsphinx',
        'sphinx-issues',
        'm2r',
        'jsonpatch',
        'ipython<7',  # jupyter_console and ipython clash in dependency requirement -- downgrade ipython for now
        'pre-commit',
        'black;python_version>="3.6"',  # Black is Python3 only
        'twine',
    ],
}
extras_require['complete'] = sorted(set(sum(extras_require.values(), [])))

setup(
    name='pyhf',
    version='0.0.16',
    description='(partial) pure python histfactory implementation',
    long_description=long_description,
    long_description_content_type='text/markdown',
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
        'jsonpatch',
    ],
    extras_require=extras_require,
    entry_points={'console_scripts': ['pyhf=pyhf.commandline:pyhf']},
    dependency_links=[],
)
