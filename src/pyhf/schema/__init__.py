"""
See :class:`~pyhf.schema.Schema` for documentation.
"""
from __future__ import annotations
import sys
from typing import Any
from pyhf.schema.loader import load_schema
from pyhf.schema.validator import validate
from pyhf.schema import variables
from pyhf.typing import Self

if sys.version_info >= (3, 9):
    from importlib import resources
else:
    import importlib_resources as resources

__all__ = [
    "load_schema",
    "validate",
    "path",
    "version",
]


def __dir__() -> list[str]:
    return __all__


class Schema(sys.modules[__name__].__class__):  # type: ignore[misc]
    """
    A module-level wrapper around :mod:`pyhf.schema` which will provide additional functionality for interacting with schemas.

    .. rubric:: Example (callable)

    .. code-block:: pycon

        >>> import pyhf.schema
        >>> import pathlib
        >>> curr_path = pyhf.schema.path
        >>> curr_path  # doctest: +ELLIPSIS
        PosixPath('.../pyhf/schemas')
        >>> new_path = pathlib.Path("/home/root/my/new/path")
        >>> pyhf.schema(new_path)  # doctest: +ELLIPSIS
        <module 'pyhf.schema' from ...>
        >>> pyhf.schema.path
        PosixPath('/home/root/my/new/path')
        >>> pyhf.schema(curr_path)  # doctest: +ELLIPSIS
        <module 'pyhf.schema' from ...>
        >>> pyhf.schema.path  # doctest: +ELLIPSIS
        PosixPath('.../pyhf/schemas')

    .. rubric:: Example (context-manager)

    .. code-block:: pycon

        >>> import pyhf.schema
        >>> import pathlib
        >>> curr_path = pyhf.schema.path
        >>> curr_path  # doctest: +ELLIPSIS
        PosixPath('.../pyhf/schemas')
        >>> new_path = pathlib.Path("/home/root/my/new/path")
        >>> with pyhf.schema(new_path):
        ...     print(repr(pyhf.schema.path))
        ...
        PosixPath('/home/root/my/new/path')
        >>> pyhf.schema.path  # doctest: +ELLIPSIS
        PosixPath('.../pyhf/schemas')

    """

    # type ignore below, see https://github.com/python/mypy/pull/11666
    def __call__(self, new_path: resources.abc.Traversable) -> Self:  # type: ignore[valid-type]
        """
        Change the local search path for finding schemas locally.

        Args:
            new_path (pathlib.Path): Path to folder containing the schemas

        Returns:
            self (pyhf.schema.Schema): Returns itself (for contextlib management)
        """
        self.orig_path, variables.schemas = variables.schemas, new_path
        self.orig_cache = dict(variables.SCHEMA_CACHE)
        variables.SCHEMA_CACHE.clear()
        return self

    def __enter__(self) -> None:
        pass

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        """
        Reset the local search path for finding schemas locally.

        Returns:
            None
        """
        variables.schemas = self.orig_path
        variables.SCHEMA_CACHE = self.orig_cache

    @property
    def path(self) -> resources.abc.Traversable:
        """
        The local path for schemas.
        """
        return variables.schemas

    @property
    def version(self) -> str:
        """
        The default version used for finding schemas.
        """
        return variables.SCHEMA_VERSION


sys.modules[__name__].__class__ = Schema
