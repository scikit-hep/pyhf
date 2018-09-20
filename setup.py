from setuptools import setup, find_packages
setup(
  name = 'pyhf',
  version = '0.0.15',
  description = '(partial) pure python histfactory implementation',
  url = '',
  author = 'Lukas Heinrich',
  author_email = 'lukas.heinrich@cern.ch',
  packages = find_packages(),
  include_package_data = True,
  install_requires = [
    'scipy',  # requires numpy, which is required by pyhf, tensorflow, and mxnet
    'click>=6.0',  # for console scripts,
    'tqdm',  # for readxml
    'six',  # for modifiers
    'jsonschema>=v3.0.0a2',  # for utils, alpha-release for draft 6
    'jsonpatch'
  ],
  extras_require = {
    'xmlimport': [
       'uproot',
     ],
    'torch': [
      'torch>=0.4.0'
    ],
    'minuit': [
      'iminuit'
    ],
    'mxnet':[
      'mxnet>=1.0.0',
      'requests<2.19.0,>=2.18.4',
      'numpy<1.15.0,>=1.8.2',
      'requests<2.19.0,>=2.18.4',
    ],
    'tensorflow':[
       'tensorflow>=1.10.0',
       'numpy<=1.14.5,>=1.14.0',  # Lower of 1.14.0 instead of 1.13.3 to ensure doctest pass
       'tensorflow-probability>=0.3.0',
       'setuptools<=39.1.0',
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
       'uproot',
       'papermill',
       'graphviz',
       'sphinx',
       'sphinxcontrib-bibtex',
       'sphinxcontrib-napoleon',
       'sphinx_rtd_theme',
       'nbsphinx',
       'm2r',
       'jsonpatch'
    ]
  },
  entry_points = {
      'console_scripts': ['pyhf=pyhf.commandline:pyhf']
  },
  dependency_links = [
  ]
)
