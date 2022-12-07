from __future__ import annotations
import sys
from pyhf.typing import Schema, SchemaVersion, Traversable

# importlib.resources.as_file wasn't added until Python 3.9
# c.f. https://docs.python.org/3.9/library/importlib.html#importlib.resources.as_file
if sys.version_info >= (3, 9):
    from importlib import resources
else:
    import importlib_resources as resources
schemas: Traversable = resources.files('pyhf') / "schemas"

SCHEMA_CACHE: dict[str, Schema] = {}
SCHEMA_BASE = "https://scikit-hep.org/pyhf/schemas/"
SCHEMA_VERSION: dict[str, SchemaVersion] = {
    'model.json': '1.0.0',
    'workspace.json': '1.0.0',
    'patchset.json': '1.0.0',
}
