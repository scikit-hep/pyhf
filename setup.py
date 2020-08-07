from setuptools import setup

extras_require = {
    'shellcomplete': ['click_completion'],
    'tensorflow': [
        'tensorflow~=2.2.0',  # TensorFlow minor releases are as volatile as major
        'tensorflow-probability~=0.10.0',
    ],
    'torch': ['torch~=1.2'],
    'jax': ['jax~=0.1,>0.1.51', 'jaxlib~=0.1,>0.1.33'],
    'xmlio': ['uproot~=3.6'],  # Future proof against uproot4 API changes
    'minuit': ['iminuit~=1.4,>=1.4.3'],  # Use "name" keyword in MINUIT optimizer
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
extras_require['lint'] = sorted(set(['pyflakes', 'black']))

extras_require['test'] = sorted(
    set(
        extras_require['backends']
        + extras_require['xmlio']
        + extras_require['contrib']
        + extras_require['shellcomplete']
        + [
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
            'jupyter',
            'uproot~=3.3',
            'graphviz',
            'jsonpatch',
        ]
    )
)
extras_require['docs'] = sorted(
    set(
        [
            'sphinx>=3.1.2',
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
        + extras_require['lint']
        + extras_require['test']
        + ['nbdime', 'bumpversion', 'ipython', 'pre-commit', 'check-manifest', 'twine']
    )
)
extras_require['complete'] = sorted(set(sum(extras_require.values(), [])))


setup(
    extras_require=extras_require,
    use_scm_version=lambda: {'local_scheme': lambda version: ''},
)
