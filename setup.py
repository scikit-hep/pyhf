from setuptools import setup

extras_require = {
    'shellcomplete': ['click_completion'],
    'tensorflow': [
        'tensorflow~=2.3,!=2.3.0',  # c.f. https://github.com/tensorflow/tensorflow/pull/40789
        'tensorflow-probability~=0.11',
    ],
    'torch': ['torch~=1.10'],
    'jax': ['jax>=0.2.10', 'jaxlib>=0.1.60,!=0.1.68'],  # c.f. Issue 1501
    'xmlio': ['uproot>=4.1.1'],
    'minuit': ['iminuit>=2.4'],
}
extras_require['backends'] = sorted(
    set(
        extras_require['tensorflow']
        + extras_require['torch']
        + extras_require['jax']
        + extras_require['minuit']
    )
)
extras_require['contrib'] = sorted({'matplotlib', 'requests'})
extras_require['lint'] = sorted({'flake8', 'black'})

extras_require['test'] = sorted(
    set(
        extras_require['backends']
        + extras_require['xmlio']
        + extras_require['contrib']
        + extras_require['shellcomplete']
        + [
            'pytest~=6.0',
            'pytest-cov>=2.5.1',
            'pytest-mock',
            'pytest-benchmark[histogram]',
            'pytest-console-scripts',
            'pytest-mpl',
            'pydocstyle',
            'papermill~=2.0',
            'nteract-scrapbook~=0.2',
            'jupyter',
            'graphviz',
        ]
    )
)
extras_require['docs'] = sorted(
    set(
        extras_require['xmlio']
        + extras_require['contrib']
        + [
            'sphinx>=4.0.0',
            'sphinxcontrib-bibtex~=2.1',
            'sphinx-click',
            'sphinx_rtd_theme',
            'nbsphinx',
            'ipywidgets',
            'sphinx-issues',
            'sphinx-copybutton>=0.3.2',
        ]
    )
)
extras_require['develop'] = sorted(
    set(
        extras_require['docs']
        + extras_require['lint']
        + extras_require['test']
        + [
            'nbdime',
            'bump2version',
            'ipython',
            'pre-commit',
            'check-manifest',
            'codemetapy>=0.3.4',
            'twine',
        ]
    )
)
extras_require['complete'] = sorted(set(sum(extras_require.values(), [])))


setup(
    extras_require=extras_require,
    use_scm_version=lambda: {'local_scheme': lambda version: ''},
)
