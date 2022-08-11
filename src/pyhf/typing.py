import os
import sys
import typing as T

if sys.version_info >= (3, 8):
    from typing import TypedDict, Literal
else:
    from typing_extensions import TypedDict, Literal

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
    "Observation",
    "Workspace",
)

if T.TYPE_CHECKING:
    PathOrStr = T.Union[str, os.PathLike[str]]
else:
    PathOrStr = T.Union[str, "os.PathLike[str]"]


class ParameterBase(TypedDict, total=False):
    auxdata: T.Sequence[float]
    bounds: T.Sequence[T.Sequence[float]]
    inits: T.Sequence[float]
    sigmas: T.Sequence[float]
    fixed: bool


class Parameter(ParameterBase):
    name: str


class Config(TypedDict):
    poi: str
    parameters: T.MutableSequence[Parameter]


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
    lo_data: T.Sequence[float]
    hi_data: T.Sequence[float]


class HistoSys(ModifierBase):
    type: Literal['histosys']
    data: HistoSysData


class StatError(ModifierBase):
    type: Literal['staterror']
    data: T.Sequence[float]


class ShapeSys(ModifierBase):
    type: Literal['shapesys']
    data: T.Sequence[float]


class ShapeFactor(ModifierBase):
    type: Literal['shapefactor']
    data: None


class LumiSys(TypedDict):
    name: Literal['lumi']
    type: Literal['lumi']
    data: None


Modifier = T.Union[
    NormSys, NormFactor, HistoSys, StatError, ShapeSys, ShapeFactor, LumiSys
]


class SampleBase(TypedDict, total=False):
    parameter_configs: T.Sequence[Parameter]


class Sample(SampleBase):
    name: str
    data: T.Sequence[float]
    modifiers: T.Sequence[Modifier]


class Channel(TypedDict):
    name: str
    samples: T.Sequence[Sample]


class Observation(TypedDict):
    name: str
    data: T.Sequence[float]


class Workspace(TypedDict):
    measurements: T.Sequence[Measurement]
    channels: T.Sequence[Channel]
    observations: T.Sequence[Observation]
