import os
import sys
from typing import MutableSequence, Sequence, Union

if sys.version_info >= (3, 8):
    from typing import Literal, TypedDict
else:
    from typing_extensions import Literal, TypedDict

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

# TODO: Switch to os.PathLike[str] once Python 3.8 support dropped
PathOrStr = Union[str, "os.PathLike[str]"]


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
