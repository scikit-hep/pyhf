from .histosys import histosys_builder, histosys_combined
from .lumi import lumi_builder, lumi_combined
from .normfactor import normfactor_builder, normfactor_combined
from .shapefactor import shapefactor_builder, shapefactor_combined
from .normsys import normsys_builder, normsys_combined
from .shapesys import shapesys_builder, shapesys_combined
from .staterror import staterr_builder, staterror_combined

pyhfset = {
    'histosys': (histosys_builder, histosys_combined),
    'lumi': (lumi_builder, lumi_combined),
    'normfactor': (normfactor_builder, normfactor_combined),
    'normsys': (normsys_builder, normsys_combined),
    'shapefactor': (shapefactor_builder, shapefactor_combined),
    'shapesys': (shapesys_builder, shapesys_combined),
    'staterror': (staterr_builder, staterror_combined),
}
