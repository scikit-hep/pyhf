from pyhf.modifiers.histosys import histosys_builder, histosys_combined
from pyhf.modifiers.lumi import lumi_builder, lumi_combined
from pyhf.modifiers.normfactor import normfactor_builder, normfactor_combined
from pyhf.modifiers.normsys import normsys_builder, normsys_combined
from pyhf.modifiers.shapefactor import shapefactor_builder, shapefactor_combined
from pyhf.modifiers.shapesys import shapesys_builder, shapesys_combined
from pyhf.modifiers.staterror import staterror_builder, staterror_combined

__all__ = [
    "histfactory_set",
    "histosys",
    "histosys_builder",
    "histosys_combined",
    "lumi",
    "lumi_builder",
    "lumi_combined",
    "normfactor",
    "normfactor_builder",
    "normfactor_combined",
    "normsys",
    "normsys_builder",
    "normsys_combined",
    "shapefactor",
    "shapefactor_builder",
    "shapefactor_combined",
    "shapesys",
    "shapesys_builder",
    "shapesys_combined",
    "staterror",
    "staterror_builder",
    "staterror_combined",
]


def __dir__():
    return __all__


histfactory_set = {
    "histosys": (histosys_builder, histosys_combined),
    "lumi": (lumi_builder, lumi_combined),
    "normfactor": (normfactor_builder, normfactor_combined),
    "normsys": (normsys_builder, normsys_combined),
    "shapefactor": (shapefactor_builder, shapefactor_combined),
    "shapesys": (shapesys_builder, shapesys_combined),
    "staterror": (staterror_builder, staterror_combined),
}
