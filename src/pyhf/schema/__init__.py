import pathlib
import sys
from pyhf.schema.loader import load_schema
from pyhf.schema.validator import validate
from pyhf.schema import variables

__all__ = [
    "load_schema",
    "validate",
    "path",
    "version",
]


def __dir__():
    return __all__


class Schema(sys.modules[__name__].__class__):
    """
    A module-level wrapper around ``pyhf.schema`` which will provide additional functionality for interacting with schemas.

    Example:
        >>> import pyhf.schema
        >>> import pathlib
        >>> curr_path = pyhf.schema.path
        >>> curr_path # doctest: +ELLIPSIS
        PosixPath('.../pyhf/schemas')
        >>> pyhf.schema(pathlib.Path('/home/root/my/new/path'))
        >>> pyhf.schema.path
        PosixPath('/home/root/my/new/path')
        >>> pyhf.schema(curr_path)
        >>> pyhf.schema.path # doctest: +ELLIPSIS
        PosixPath('.../pyhf/schemas')

    """

    def __call__(self, new_path: pathlib.Path):
        """
        Change the local search path for finding schemas locally.

        Args:
            new_path (pathlib.Path): Path to folder containing the schemas

        Returns:
            None
        """
        variables.schemas = new_path

    @property
    def path(self):
        """
        The local path for schemas.
        """
        return variables.schemas

    @property
    def version(self):
        """
        The default version used for finding schemas.
        """
        return variables.SCHEMA_VERSION


sys.modules[__name__].__class__ = Schema
