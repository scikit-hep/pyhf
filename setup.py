from setuptools import setup

extras_require = {
    'shellcomplete': ['click_completion'],
    'tensorflow': [
        'tensorflow>=2.3.1',  # c.f. https://github.com/tensorflow/tensorflow/pull/40789
        'tensorflow-probability>=0.11.0',  # c.f. PR #1657
    ],
    'torch': ['torch>=1.10.0'],  # c.f. PR #1657
    'jax': ['jax>=0.2.10', 'jaxlib>=0.1.60,!=0.1.68'],  # c.f. Issue 1501
    'xmlio': ['uproot>=4.1.1'],  # c.f. PR #1567
    'minuit': ['iminuit>=2.4.0'],  # c.f. PR #1306
}
extras_require['backends'] = sorted(
    set(
        extras_require['tensorflow']
        + extras_require['torch']
        + extras_require['jax']
        + extras_require['minuit']
    )
)
extras_require['contrib'] = sorted({'matplotlib', 'requests', 'rich-click>=1.2.0'})
extras_require['lint'] = sorted({'flake8', 'black>=22.1.0'})

extras_require['test'] = sorted(
    set(
        extras_require['backends']
        + extras_require['xmlio']
        + extras_require['contrib']
        + extras_require['shellcomplete']
        + [
            'scikit-hep-testdata>=0.4.11',
            'pytest>=6.0',
            'pytest-cov>=2.5.1',
            'pytest-mock',
            'requests-mock>=1.9.0',
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
            'nbsphinx!=0.8.8',  # c.f. https://github.com/spatialaudio/nbsphinx/issues/620
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
