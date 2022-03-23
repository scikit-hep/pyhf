import sys

# importlib.resources.as_file wasn't added until Python 3.9
# c.f. https://docs.python.org/3.9/library/importlib.html#importlib.resources.as_file
if sys.version_info >= (3, 9):
    from importlib import resources
else:
    import importlib_resources as resources
schemas = resources.files('pyhf') / "schemas"

SCHEMA_CACHE = {}
SCHEMA_BASE = "https://scikit-hep.org/pyhf/schemas/"
SCHEMA_VERSION = '1.0.0'
