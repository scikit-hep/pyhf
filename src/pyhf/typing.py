import os
from collections.abc import MutableSequence, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Protocol,
    SupportsIndex,
    TypedDict,
)

if TYPE_CHECKING:
    from jax import Array as JaxArray

    from pyhf.tensor.numpy_backend import Tensor as NumPyTensor

    # A tensor of any of the supported backends. Only defined for type checking, as
    # jax is an optional dependency and pyhf.tensor.numpy_backend imports this module.
    Tensor = NumPyTensor | JaxArray

__all__ = (
    "Channel",
    "HistoSys",
    "Literal",
    "LumiSys",
    "Measurement",
    "Modifier",
    "ModifierBase",
    "NormFactor",
    "NormSys",
    "Observation",
    "Parameter",
    "ParameterBase",
    "PathOrStr",
    "Protocol",
    "Sample",
    "ShapeFactor",
    "ShapeSys",
    "StatError",
    "TypedDict",
    "Workspace",
)


PathOrStr = str | os.PathLike[str]

Shape = tuple[int, ...]
ShapeLike = SupportsIndex | Sequence[SupportsIndex]


class ParameterBase(TypedDict, total=False):
    auxdata: Sequence[float]
    bounds: Sequence[Sequence[float]]
    inits: Sequence[float]
    sigmas: Sequence[float]
    fixed: bool


class Parameter(ParameterBase):
    name: str


class Config(TypedDict):
    poi: str
    parameters: MutableSequence[Parameter]


class Measurement(TypedDict):
    name: str
    config: Config


class ModifierBase(TypedDict):
    name: str


class NormSysData(TypedDict):
    lo: float
    hi: float


class NormSys(ModifierBase):
    type: Literal["normsys"]
    data: NormSysData


class NormFactor(ModifierBase):
    type: Literal["normfactor"]
    data: None


class HistoSysData(TypedDict):
    lo_data: Sequence[float]
    hi_data: Sequence[float]


class HistoSys(ModifierBase):
    type: Literal["histosys"]
    data: HistoSysData


class StatError(ModifierBase):
    type: Literal["staterror"]
    data: Sequence[float]


class ShapeSys(ModifierBase):
    type: Literal["shapesys"]
    data: Sequence[float]


class ShapeFactor(ModifierBase):
    type: Literal["shapefactor"]
    data: None


class LumiSys(TypedDict):
    name: Literal["lumi"]
    type: Literal["lumi"]
    data: None


Modifier = (
    NormSys | NormFactor | HistoSys | StatError | ShapeSys | ShapeFactor | LumiSys
)


class SampleBase(TypedDict, total=False):
    parameter_configs: Sequence[Parameter]


class Sample(SampleBase):
    name: str
    data: Sequence[float]
    modifiers: Sequence[Modifier]


class Channel(TypedDict):
    name: str
    samples: Sequence[Sample]


class Observation(TypedDict):
    name: str
    data: Sequence[float]


class Workspace(TypedDict):
    measurements: Sequence[Measurement]
    channels: Sequence[Channel]
    observations: Sequence[Observation]


class TensorBackend(Protocol):
    name: str
    precision: str
    default_do_grad: bool

    def _setup(self) -> None: ...


class Optimizer(Protocol):
    name: str


class PDF(Protocol):
    def sample(self, sample_shape: Shape) -> Any: ...

    def log_prob(self, value: Any) -> Any: ...
