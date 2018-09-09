import logging
log = logging.getLogger(__name__)

from . import get_backend
from . import exceptions
from . import utils

@utils.tensorize_args
def _hfinterp_code0(at_minus_one, at_zero, at_plus_one, alphas):
    tensorlib, _ = get_backend()
    #warning: alphas must be ordered
    up_variation  = at_plus_one - at_zero
    down_variation = at_zero - at_minus_one

    positive_parameters = alphas[alphas >= 0]
    negative_parameters = alphas[alphas < 0]

    w_pos = positive_parameters * tensorlib.ones(up_variation.shape + positive_parameters.shape)
    r_pos = up_variation.reshape(up_variation.shape + (1,)) * w_pos

    w_neg = negative_parameters * tensorlib.ones(down_variation.shape + negative_parameters.shape)
    r_neg = down_variation.reshape(down_variation.shape + (1,)) * w_neg

    result = tensorlib.concatenate([r_neg, r_pos], axis=-1)
    result = tensorlib.einsum('...ij->...ji', result)
    return result

@utils.tensorize_args
def _hfinterp_code1(at_minus_one, at_zero, at_plus_one, alphas):
    tensorlib, _ = get_backend()
    #warning: alphas must be ordered
    up_variation = tensorlib.divide(at_plus_one, at_zero)
    down_variation = tensorlib.divide(at_minus_one, at_zero)

    positive_parameters = alphas[alphas >= 0]
    negative_parameters = alphas[alphas < 0]

    bases_negative = tensorlib.tile(down_variation, negative_parameters.shape+(1,)*len(down_variation.shape))
    bases_negative = tensorlib.einsum('i...->...i', bases_negative)

    bases_positive = tensorlib.tile(up_variation, positive_parameters.shape+(1,)*len(up_variation.shape))
    bases_positive = tensorlib.einsum('i...->...i', bases_positive)

    expo_positive = tensorlib.tile(positive_parameters, up_variation.shape+(1,)) #was outer
    expo_negative = -tensorlib.tile(negative_parameters, down_variation.shape+(1,)) #was outer

    res_neg = tensorlib.power(bases_negative, expo_negative)
    res_pos = tensorlib.power(bases_positive, expo_positive)

    result = tensorlib.concatenate([res_neg, res_pos], axis=-1)
    return tensorlib.einsum('...ij->...ji', result)

# interpolation codes come from https://cds.cern.ch/record/1456844/files/CERN-OPEN-2012-016.pdf
def interpolator(interpcode):
    interpcodes = {0: _hfinterp_code0,
                   1: _hfinterp_code1}
    try:
        return interpcodes[interpcode]
    except KeyError:
        raise exceptions.InvalidInterpCode
