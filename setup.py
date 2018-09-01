from setuptools import setup, find_packages
setup(
  name = 'pyhf',
  version = '0.0.13',
  description = '(partial) pure python histfactory implementation',
  url = '',
  author = 'Lukas Heinrich',
  author_email = 'lukas.heinrich@cern.ch',
  packages = find_packages(),
  include_package_data = True,
  install_requires = [
    'numpy<=1.14.5,>=1.14.3',  # required by tensorflow, mxnet, and us
    'scipy',
    'click>=6.0',  # for console scripts,
    'tqdm',  # for readxml
  ],
  extras_require = {
    'xmlimport': [
       'uproot',
     ],
    'torch': [
      'torch>=0.4.0'
    ],
    'mxnet':[
      'mxnet>=1.0.0',
      'requests<2.19.0,>=2.18.4',
      'numpy<1.15.0,>=1.8.2',
      'requests<2.19.0,>=2.18.4',
    ],
    'tensorflow':[
       'tensorflow==1.10.0',
       'numpy<=1.14.5,>=1.13.3',
       'setuptools<=39.1.0',
    ],
    'develop': [
       'pyflakes',
       'pytest>=3.5.1',
       'pytest-cov>=2.5.1',
       'pytest-benchmark[histogram]',
       'pytest-console-scripts',
       'python-coveralls',
       'coverage==4.0.3',  # coveralls
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
       'jsonpatch',
       'jsonschema==v3.0.0a2'  # alpha-release for draft 6
    ]
  },
  entry_points = {
      'console_scripts': ['pyhf=pyhf.commandline:pyhf']
  },
  dependency_links = [
  ]
)
