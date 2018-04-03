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
       # 'torch',
       "http://download.pytorch.org/whl/cpu/torch-0.3.1-cp27-cp27mu-linux_x86_64.whl; python_version=='2.7'",
       "http://download.pytorch.org/whl/cpu/torch-0.3.1-cp35-cp35m-linux_x86_64.whl; python_version=='3.5'",
       "http://download.pytorch.org/whl/cpu/torch-0.3.1-cp36-cp36m-linux_x86_64.whl; python_version=='3.6'",
       'tensorflow',
       'mxnet>=1.0.0',
       'graphviz',
       'sphinx',
       'sphinxcontrib-napoleon',
       'sphinx_rtd_theme',
       'nbsphinx',
       'jsonschema>=2.6.0'
    ]
  },
  entry_points = {
  },
  dependency_links = [
  ]
)
