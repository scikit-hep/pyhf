from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent.resolve()
with open(Path().joinpath(this_directory, 'README.md'), encoding='utf-8') as readme_md:
    long_description = readme_md.read()

extras_require = {
    'tensorflow': ['tensorflow~=2.0', 'tensorflow-probability~=0.8'],
    'torch': ['torch~=1.2'],
    'jax': ['jax~=0.1,>0.1.51', 'jaxlib~=0.1,>0.1.33'],
    'xmlio': ['uproot'],
    'minuit': ['iminuit'],
}
extras_require['backends'] = sorted(
    set(
        extras_require['tensorflow']
        + extras_require['torch']
        + extras_require['jax']
        + extras_require['minuit']
    )
)
extras_require['contrib'] = sorted(set(['matplotlib']))

extras_require['test'] = sorted(
    set(
        extras_require['backends']
        + extras_require['xmlio']
        + extras_require['contrib']
        + [
            'pyflakes',
            'pytest~=3.5',
            'pytest-cov>=2.5.1',
            'pytest-mock',
            'pytest-benchmark[histogram]',
            'pytest-console-scripts',
            'pytest-mpl',
            'pydocstyle',
            'coverage>=4.0',  # coveralls
            'papermill~=1.0',
            'nteract-scrapbook~=0.2',
            'check-manifest',
            'matplotlib',
            'jupyter',
            'uproot~=3.3',
            'graphviz',
            'jsonpatch',
            'black;python_version>="3.6"',  # Black is Python3 only
        ]
    )
)
extras_require['docs'] = sorted(
    set(
        [
            'sphinx',
            'sphinxcontrib-bibtex',
            'sphinx-click',
            'sphinx_rtd_theme',
            'nbsphinx',
            'ipywidgets',
            'sphinx-issues',
            'm2r',
        ]
    )
)
extras_require['develop'] = sorted(
    set(
        extras_require['docs']
        + extras_require['test']
        + ['nbdime', 'bumpversion', 'ipython', 'pre-commit', 'twine']
    )
)
extras_require['complete'] = sorted(set(sum(extras_require.values(), [])))


setup(
    name='pyhf',
    version='0.4.0',
    description='(partial) pure python histfactory implementation',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/scikit-hep/pyhf',
    author='Lukas Heinrich, Matthew Feickert, Giordon Stark',
    author_email='lukas.heinrich@cern.ch, matthew.feickert@cern.ch, gstark@cern.ch',
    license='Apache',
    keywords='physics fitting numpy scipy tensorflow pytorch',
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    include_package_data=True,
    python_requires=">=3.6",
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
    entry_points={'console_scripts': ['pyhf=pyhf.cli:cli']},
    dependency_links=[],
    use_scm_version=lambda: {'local_scheme': lambda version: ''},
)
