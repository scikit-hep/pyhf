from setuptools import setup, find_packages
setup(
  name = 'pyhf',
  version = '0.0.4',
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
    'develop': [
       'pyflakes',
       'pytest>=3.2.0',
       'pytest-cov>=2.5.1',
       'python-coveralls',
       'matplotlib',
       'jupyter',
       'uproot',
       'papermill',
       'torch',
       'tensorflow'
    ]
  },
  entry_points = {
  },
  dependency_links = [
  ]
)
