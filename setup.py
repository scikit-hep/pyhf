from setuptools import setup

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
            'papermill~=2.0',
            'nteract-scrapbook~=0.2',
            'check-manifest',
            'jupyter',
            'uproot~=3.3',
            'graphviz',
            'jsonpatch',
            'black',
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
            'sphinx-copybutton>0.2.9',
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
    extras_require=extras_require,
    use_scm_version=lambda: {'local_scheme': lambda version: ''},
)
