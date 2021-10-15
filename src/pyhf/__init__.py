from pyhf.tensor import BackendRetriever as tensor
from pyhf.tensor.manager import get_backend
from pyhf.tensor.manager import set_backend
from pyhf._version import version as __version__
from pyhf.exceptions import InvalidBackend, InvalidOptimizer, Unsupported

from pyhf.pdf import Model
from pyhf.workspace import Workspace
from pyhf import simplemodels
from pyhf import infer
from pyhf import compat
from pyhf.patchset import PatchSet

__all__ = [
    "Model",
    "PatchSet",
    "Workspace",
    "__version__",
    "compat",
    "default_backend",
    "exceptions",
    "get_backend",
    "infer",
    "interpolators",
    "modifiers",
    "optimizer",
    "parameters",
    "patchset",
    "pdf",
    "probability",
    "set_backend",
    "simplemodels",
    "tensor",
    "tensorlib",
    "utils",
    "workspace",
]


def __dir__():
    return __all__


def __getattr__(name):
    if name == 'tensorlib':
        return get_backend(default=False)[0]
    elif name == 'optimizer':
        return get_backend(default=False)[1]
    elif name == 'default_backend':
        return get_backend(default=True)[0]
