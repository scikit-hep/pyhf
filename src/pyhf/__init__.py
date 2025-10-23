# FIXME: If import order is changed 'import pyhf' fails due to circular imports
# ruff: isort: off
from pyhf._version import version as __version__
from pyhf.tensor import BackendRetriever as tensor
from pyhf.optimize import OptimizerRetriever as optimize  # noqa
from pyhf.tensor.manager import get_backend, set_backend
from pyhf.pdf import Model
from pyhf.workspace import Workspace
from pyhf import schema, simplemodels, infer, compat
from pyhf.patchset import PatchSet
# ruff: isort: on

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
    "schema",
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
    if name == "tensorlib":
        return get_backend(default=False)[0]
    if name == "optimizer":
        return get_backend(default=False)[1]
    if name == "default_backend":
        return get_backend(default=True)[0]
    raise AttributeError
