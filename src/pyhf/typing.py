import sys

if sys.version_info >= (3, 8):
    from typing import Protocol
else:
    from typing_extensions import Protocol

from typing import Any, Iterable, Callable, Sequence, Union, Tuple

Shape = Tuple[int, ...]
Tensor = Any
FloatOrTensor = Union[float, Tensor]


class PDF(Protocol):
    def expected_data(self) -> Tensor:
        ...

    def log_prob(self, value: float) -> float:
        ...


class TensorBackend(Protocol):
    def __init__(self, **kwargs: Any) -> None:
        ...

    def _setup(self) -> None:
        ...

    def abs(self, tensor_in: Tensor) -> Tensor:
        ...

    def astensor(self, tensor_in: Tensor, _: str) -> Tensor:
        ...

    def boolean_mask(self, tensor_in: Tensor, mask: Sequence[bool]) -> Tensor:
        ...

    def clip(self, tensor_in: Tensor, minval: float, maxval: float) -> Tensor:
        ...

    def concatenate(self, sequence: Sequence, axis: int) -> Tensor:
        ...

    def conditional(
        self,
        predicate: bool,
        call_true: Callable[[], bool],
        call_false: Callable[[], bool],
    ) -> Tensor:
        ...

    def divide(self, tensor_in_1: Tensor, tensor_in_2: Tensor) -> Tensor:
        ...

    def einsum(self, subscripts: str, *operands: Tensor) -> Tensor:
        ...

    def exp(self, tensor_in: Tensor) -> Tensor:
        ...

    def gather(self, tensor_in: Tensor, indices: Sequence[int]) -> Tensor:
        ...

    def isfinite(self, tensor_in: Tensor) -> Tensor:
        ...

    def log(self, tensor_in: Tensor) -> Tensor:
        ...

    def normal(
        self, x: FloatOrTensor, mu: FloatOrTensor, sigma: FloatOrTensor
    ) -> Tensor:
        ...

    def normal_cdf(
        self, x: FloatOrTensor, mu: FloatOrTensor, sigma: FloatOrTensor
    ) -> Tensor:
        ...

    def normal_dist(self, mu: FloatOrTensor, sigma: FloatOrTensor) -> PDF:
        ...

    def normal_logpdf(
        self, x: FloatOrTensor, mu: FloatOrTensor, sigma: FloatOrTensor
    ) -> Tensor:
        ...

    def ones(self, shape: Union[int, Iterable]) -> Tensor:
        ...

    def outer(self, tensor_in_1: Tensor, tensor_in_2: Tensor) -> Tensor:
        ...

    def poisson(self, n: FloatOrTensor, lam: FloatOrTensor) -> FloatOrTensor:
        ...

    def poisson_dist(self, rate: FloatOrTensor) -> PDF:
        ...

    def poisson_logpdf(self, n: FloatOrTensor, lam: FloatOrTensor) -> FloatOrTensor:
        ...

    def power(self, tensor_in_1: Tensor, tensor_in_2: Tensor) -> Tensor:
        ...

    def product(self, tensor_in: Tensor, axis: int) -> Tensor:
        ...

    def reshape(self, tensor_in: Tensor, newshape: Shape) -> Tensor:
        ...

    def shape(self, tensor_in: Tensor) -> Shape:
        ...

    def simple_broadcast(self, *args: Tensor) -> Tensor:
        ...

    def sqrt(self, tensor_in: Tensor) -> Tensor:
        ...

    def stack(self, sequence: Sequence, axis: int) -> Tensor:
        ...

    def sum(self, tensor_in: Tensor, axis: int) -> Tensor:
        ...

    def tensor(self):
        ...

    def tile(self, tensor_in: Tensor, repeats: Tensor):
        ...

    def tolist(self, tensor_in: Tensor) -> Sequence:
        ...

    def where(self, mask: Tensor, tensor_in_1: Tensor, tensor_in_2: Tensor) -> Tensor:
        ...

    def zeros(self, shape: Union[int, Iterable]) -> Tensor:
        ...
