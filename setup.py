from setuptools import setup, find_packages
setup(
  name = 'pyhf',
  version = '0.0.8',
  description = '(partial) pure python histfactory implementation',
  url = '',
  author = 'Lukas Heinrich',
  author_email = 'lukas.heinrich@cern.ch',
  packages = find_packages(),
  include_package_data = True,
  install_requires = [
    'numpy',
    'scipy'
  ],
  extras_require = {
    'xmlimport': [
       'uproot',
     ],
    'torch': [
      'torch'
    ],
    'mxnet':[
      'mxnet',
    ],
    'develop': [
       'pyflakes',
       'pytest>=3.2.0',
       'pytest-cov>=2.5.1',
       'pytest-benchmark[histogram]',
       'python-coveralls',
       'matplotlib',
       'jupyter',
       'uproot',
       'papermill',
       'torch',
       'tensorflow',
       'mxnet>=1.0.0',
       'graphviz',
       'sphinx',
       'jsonschema>=2.6.0'
    ]
  },
  entry_points = {
  },
  dependency_links = [
  ]
)
