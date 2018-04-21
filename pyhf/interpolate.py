import logging
log = logging.getLogger(__name__)

from . import get_backend
from . import exceptions

def _hfinterp_code0(at_minus_one, at_zero, at_plus_one, alphas):
    tensorlib, _ = get_backend()
    at_minus_one = tensorlib.astensor(at_minus_one)
    at_zero = tensorlib.astensor(at_zero)
    at_plus_one = tensorlib.astensor(at_plus_one)
    alphas = tensorlib.astensor(alphas)

    iplus_izero  = at_plus_one - at_zero
    izero_iminus = at_zero - at_minus_one
    mask = tensorlib.outer(alphas < 0, tensorlib.ones(iplus_izero.shape))
    return tensorlib.where(mask, tensorlib.outer(alphas, izero_iminus), tensorlib.outer(alphas, iplus_izero))

def _hfinterp_code1(at_minus_one, at_zero, at_plus_one, alphas):
    tensorlib, _ = get_backend()
    at_minus_one = tensorlib.astensor(at_minus_one)
    at_zero = tensorlib.astensor(at_zero)
    at_plus_one = tensorlib.astensor(at_plus_one)
    alphas = tensorlib.astensor(alphas)

    base_positive = tensorlib.divide(at_plus_one,  at_zero)
    base_negative = tensorlib.divide(at_minus_one, at_zero)
    expo_positive = tensorlib.outer(alphas, tensorlib.ones(base_positive.shape))
    mask = tensorlib.outer(alphas > 0, tensorlib.ones(base_positive.shape))
    bases = tensorlib.where(mask,base_positive,base_negative)
    exponents = tensorlib.where(mask, expo_positive,-expo_positive)
    return tensorlib.power(bases, exponents)

# interpolation codes come from https://cds.cern.ch/record/1456844/files/CERN-OPEN-2012-016.pdf
def interpolator(interpcode):
    interpcodes = {0: _hfinterp_code0,
                   1: _hfinterp_code1}
    try:
        return interpcodes[interpcode]
    except KeyError:
        raise exceptions.InvalidInterpCode
