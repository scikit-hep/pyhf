import os

import typing as T
from typing_extensions import TypedDict  # for python 3.7 only (3.8+ has T.TypedDict)

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
    auxdata: list[float]
    bounds: list[list[float]]
    inits: list[float]
    sigmas: list[float]
    fixed: bool


class Parameter(ParameterBase):
    name: str


class Config(TypedDict):
    poi: str
    parameters: list[Parameter]


class Measurement(TypedDict):
    name: str
    config: Config


class ModifierBase(TypedDict):
    name: str


class NormSysData(TypedDict):
    lo: float
    hi: float


class NormSys(ModifierBase):
    type: T.Literal['normsys']
    data: NormSysData


class NormFactor(ModifierBase):
    type: T.Literal['normfactor']
    data: None


class HistoSysData(TypedDict):
    lo_data: list[float]
    hi_data: list[float]


class HistoSys(ModifierBase):
    type: T.Literal['histosys']
    data: HistoSysData


class StatError(ModifierBase):
    type: T.Literal['staterror']
    data: list[float]


class ShapeSys(ModifierBase):
    type: T.Literal['shapesys']
    data: list[float]


class ShapeFactor(ModifierBase):
    type: T.Literal['shapefactor']
    data: None


class LumiSys(TypedDict):
    name: T.Literal['lumi']
    type: T.Literal['lumi']
    data: None


Modifier = T.Union[
    NormSys, NormFactor, HistoSys, StatError, ShapeSys, ShapeFactor, LumiSys
]


class SampleBase(TypedDict, total=False):
    parameter_configs: list[Parameter]


class Sample(SampleBase):
    name: str
    data: list[float]
    modifiers: list[Modifier]


class Channel(TypedDict):
    name: str
    samples: list[Sample]


class Observation(TypedDict):
    name: str
    data: list[float]


class Workspace(TypedDict):
    measurements: list[Measurement]
    channels: list[Channel]
    observations: list[Observation]
