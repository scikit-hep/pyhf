"""Define pyhf version information."""

# Use semantic versioning (https://semver.org/)
_MAJOR_VERSION = '0'
_MINOR_VERSION = '0'
_PATCH_VERSION = '16'

# When making a PyPI release change to _VERSION_SUFFIX = '' to reflect a
# stable release. Otherwise, versions are in developenment phase.
_VERSION_SUFFIX = 'dev'

# Prevent trailing dot when version suffix is empty
__version__ = '.'.join(s for s in [
    _MAJOR_VERSION,
    _MINOR_VERSION,
    _PATCH_VERSION,
    _VERSION_SUFFIX,
] if s)
