from importlib import resources

schemas = resources.files('pyhf') / "schemas"

SCHEMA_CACHE = {}
SCHEMA_BASE = "https://scikit-hep.org/pyhf/schemas/"
SCHEMA_VERSION = '1.0.0'
