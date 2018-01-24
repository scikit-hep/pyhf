from setuptools import setup, find_packages
setup(
  name = 'pyhf',
  version = '0.0.1',
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
  },
  entry_points = {
  },
  dependency_links = [
  ]
)
