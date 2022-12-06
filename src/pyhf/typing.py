import os
import sys
from typing import (
    Any,
    Literal,
    Mapping,
    MutableSequence,
    Protocol,
    Sequence,
    SupportsFloat as Numeric,
    SupportsIndex,
    Tuple,
    TypedDict,
    Union,
)

if sys.version_info >= (3, 9):
    from importlib.abc import Traversable
else:
    from importlib_resources.abc import Traversable

if sys.version_info >= (3, 11):
    from typing import NotRequired, Self
else:
    from typing_extensions import NotRequired, Self

__all__ = (
    "PathOrStr",
    "ParameterBase",
    "Parameter",
    "Measurement",
    "ModifierBase",
    "NormSys",
    "NormFactor",
    "HistoSys",
    "StatError",
    "ShapeSys",
    "ShapeFactor",
    "LumiSys",
    "Modifier",
    "Sample",
    "Channel",
    "Model",
    "Observation",
    "PatchSet",
    "Workspace",
    "Literal",
    "Protocol",
    "Self",
    "Traversable",
    "TypedDict",
)


# TODO: Switch to os.PathLike[str] once Python 3.8 support dropped
PathOrStr = Union[str, "os.PathLike[str]"]

Shape = Tuple[int, ...]
ShapeLike = Union[SupportsIndex, Sequence[SupportsIndex]]

Schema = Mapping[str, Any]
SchemaVersion = Literal['1.0.1', '1.0.0']


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
    type: Literal['normsys']
    data: NormSysData


class NormFactor(ModifierBase):
    type: Literal['normfactor']
    data: None


class HistoSysData(TypedDict):
    lo_data: Sequence[float]
    hi_data: Sequence[float]


class HistoSys(ModifierBase):
    type: Literal['histosys']
    data: HistoSysData


class StatError(ModifierBase):
    type: Literal['staterror']
    data: Sequence[float]


class ShapeSys(ModifierBase):
    type: Literal['shapesys']
    data: Sequence[float]


class ShapeFactor(ModifierBase):
    type: Literal['shapefactor']
    data: None


class LumiSys(TypedDict):
    name: Literal['lumi']
    type: Literal['lumi']
    data: None


Modifier = Union[
    NormSys, NormFactor, HistoSys, StatError, ShapeSys, ShapeFactor, LumiSys
]


class Sample(TypedDict):
    name: str
    data: Sequence[float]
    modifiers: Sequence[Modifier]
    parameter_configs: NotRequired[Sequence[Parameter]]


class Channel(TypedDict):
    name: str
    samples: Sequence[Sample]


class Observation(TypedDict):
    name: str
    data: Sequence[float]


class Model(TypedDict):
    channels: Sequence[Channel]
    parameters: NotRequired[Sequence[Parameter]]


class Workspace(TypedDict):
    measurements: Sequence[Measurement]
    channels: Sequence[Channel]
    observations: Sequence[Observation]
    version: SchemaVersion


class PatchMetadata(TypedDict):
    name: str
    values: Sequence[Union[Numeric, str]]


class Patch(TypedDict):
    patch: Sequence[Mapping[str, Any]]
    metadata: PatchMetadata


class PatchSetMetadata(TypedDict):
    digests: Mapping[str, str]
    labels: Sequence[str]
    description: str
    references: Mapping[str, str]


class PatchSet(TypedDict):
    patches: Sequence[Patch]
    metadata: PatchSetMetadata
    version: SchemaVersion


class TensorBackend(Protocol):
    name: str
    precision: str
    default_do_grad: bool

    def _setup(self) -> None:
        ...


class Optimizer(Protocol):
    name: str


class PDF(Protocol):
    def sample(self, sample_shape: Shape) -> Any:
        ...

    def log_prob(self, value: Any) -> Any:
        ...
