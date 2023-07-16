"""Array API Tensor Library Module."""
from __future__ import annotations

import array_api_compat

from pyhf.tensor.manager import get_backend

# FIXME: Make work on set_backend call by moving this out of import level executed code
tensorlib, _ = get_backend()
xp = array_api_compat.array_namespace(tensorlib.astensor([]))


class array_backend:
    """Array API backend for pyhf"""

    __slots__ = ["name", "precision", "dtypemap", "default_do_grad"]

    #: The array type
    array_type = type(xp.tensor([]))

    #: The array content type
    array_subtype = type(xp.tensor([]))

    def __init__(self, **kwargs):
        self.name = xp.__name__.split(".")[-1]
        self.precision = kwargs.get("precision", "64b")
        self.dtypemap = {
            "float": xp.float64 if self.precision == "64b" else xp.float32,
            "int": xp.int64 if self.precision == "64b" else xp.int32,
            "bool": xp.bool,
        }
        self.default_do_grad = xp.is_grad_enabled()

    def _setup(self):
        """
        Run any global setups for the array lib.
        """

    def astensor(self, array_in, dtype="float"):
        """
        Convert to a PyTorch Tensor.

        Example:

            >>> import pyhf
            >>> pyhf.set_backend("pytorch")
            >>> pyhf.set_backend("array")
            >>> tensor = pyhf.tensorlib.astensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            >>> tensor
            tensor([[1., 2., 3.],
                    [4., 5., 6.]])
            >>> type(tensor)
            <class "torch.Tensor">

        Args:
            tensor_in (Number or Tensor): Tensor object

        Returns:
            torch.Tensor: A multi-dimensional matrix containing elements of a single data type.
        """
        try:
            dtype = self.dtypemap[dtype]
        except KeyError:
            log.error(
                "Invalid dtype: dtype must be float, int, or bool.", exc_info=True
            )
            raise

        return xp.as_tensor(array_in, dtype=dtype)
