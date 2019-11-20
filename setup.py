from setuptools import setup, find_packages
from os import path
import sys

this_directory = path.abspath(path.dirname(__file__))
if sys.version_info.major < 3:
    from io import open
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as readme_md:
    long_description = readme_md.read()

extras_require = {
    'tensorflow': ['tensorflow~=1.15', 'tensorflow-probability~=0.8', 'numpy~=1.16'],
    'torch': ['torch~=1.2'],
    'xmlio': ['uproot'],
    'minuit': ['iminuit'],
    'develop': [
        'pyflakes',
        'pytest~=3.5',
        'pytest-cov>=2.5.1',
        'pytest-mock',
        'pytest-benchmark[histogram]',
        'pytest-console-scripts',
        'pydocstyle',
        'coverage>=4.0',  # coveralls
        'matplotlib',
        'jupyter',
        'nbdime',
        'uproot~=3.3',
        'papermill~=1.0',
        'nteract-scrapbook~=0.2',
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
        'ipython',
        'pre-commit',
        'black;python_version>="3.6"',  # Black is Python3 only
        'twine',
        'check-manifest',
    ],
}
extras_require['complete'] = sorted(set(sum(extras_require.values(), [])))


def _is_test_pypi():
    """
    Determine if the CI environment has IS_COMMIT_TAGGED defined and
    set to true (c.f. .github/workflows/publish-package-to-pypi.yml)
    The use_scm_version kwarg accepts a callable for the local_scheme
    configuration parameter with argument "version". This can be replaced
    with a lambda as the desired version structure is {next_version}.dev{distance}
    c.f. https://github.com/pypa/setuptools_scm/#importing-in-setuppy
    As the scm versioning is only desired for TestPyPI, for depolyment to PyPI the version
    controlled through bumpversion is used.
    """
    from os import getenv

    return (
        {'local_scheme': lambda version: ''}
        if getenv('IS_COMMIT_TAGGED') == 'false'
        else False
    )


setup(
    name='pyhf',
    version='0.2.2',
    description='(partial) pure python histfactory implementation',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/scikit-hep/pyhf',
    author='Lukas Heinrich, Matthew Feickert, Giordon Stark',
    author_email='lukas.heinrich@cern.ch, matthew.feickert@cern.ch, gstark@cern.ch',
    license='Apache',
    keywords='physics fitting numpy scipy tensorflow pytorch',
    classifiers=[
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    include_package_data=True,
    python_requires=">=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*, !=3.5.*",
    install_requires=[
        'scipy',  # requires numpy, which is required by pyhf and tensorflow
        'click>=6.0',  # for console scripts,
        'tqdm',  # for readxml
        'six',  # for modifiers
        'jsonschema>=v3.0.0a2',  # for utils, alpha-release for draft 6
        'jsonpatch',
        'pyyaml',  # for parsing CLI equal-delimited options
    ],
    extras_require=extras_require,
    entry_points={'console_scripts': ['pyhf=pyhf.commandline:pyhf']},
    dependency_links=[],
    use_scm_version=_is_test_pypi(),
)
