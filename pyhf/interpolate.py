import logging
log = logging.getLogger(__name__)

from . import get_backend
from . import exceptions
from . import utils

def _kitchensink_looper(histogramssets, alphasets, func):
    all_results = []
    for histoset, alphaset in zip(histogramssets, alphasets):
        all_results.append([])
        set_result = all_results[-1]
        for histo in histoset:
            set_result.append([])
            histo_result = set_result[-1]
            for alpha in alphaset:
                alpha_result = []
                for down,nom,up in zip(histo[0],histo[1],histo[2]):
                    v = func(down, nom, up, alpha)
                    alpha_result.append(v)
                histo_result.append(alpha_result)
    return all_results

@utils.tensorize_args
def _hfinterp_code0(histogramssets, alphasets):
    tensorlib, _ = get_backend()

    allset_all_histo_deltas_up = histogramssets[:,:,2] - histogramssets[:,:,1]
    allset_all_histo_deltas_dn = histogramssets[:,:,1] - histogramssets[:,:,0]
    allset_all_histo_nom = histogramssets[:,:,1]

    #x is dummy index
    allsets_all_histos_alphas_times_deltas_up = tensorlib.einsum('sa,sxu->sxau',alphasets,allset_all_histo_deltas_up)
    allsets_all_histos_alphas_times_deltas_dn = tensorlib.einsum('sa,sxu->sxau',alphasets,allset_all_histo_deltas_dn)
    where_alphasets_positive = tensorlib.where(alphasets > 0, tensorlib.ones(alphasets.shape), tensorlib.zeros(alphasets.shape))
    allsets_all_histos_masks = tensorlib.einsum('sa,sxu->sxau', where_alphasets_positive, tensorlib.ones(allset_all_histo_deltas_dn.shape))

    allsets_all_histos_deltas = tensorlib.where(allsets_all_histos_masks,allsets_all_histos_alphas_times_deltas_up, allsets_all_histos_alphas_times_deltas_dn)
    allsets_all_histos_noms_repeated = tensorlib.einsum('sa,sxu->sxau',tensorlib.ones(alphasets.shape),allset_all_histo_nom)
    set_results = allsets_all_histos_deltas + allsets_all_histos_noms_repeated
    return set_results

def _kitchensink_code0(histogramssets, alphasets):
    def summand(down, nom, up, alpha):
        delta_up = up - nom
        delta_down = nom - down
        if alpha > 0:
            delta =  delta_up*alpha
        else:
            delta =  delta_down*alpha
        return nom + delta

    return _kitchensink_looper(histogramssets, alphasets, summand)

@utils.tensorize_args
def _hfinterp_code1(histogramssets, alphasets):
    tensorlib, _ = get_backend()
    #warning: alphasets must be ordered
    at_plus_one, at_zero, at_minus_one = histogramssets[0][0]

    up_variation = tensorlib.divide(at_plus_one, at_zero)
    down_variation = tensorlib.divide(at_minus_one, at_zero)

    positive_parameters = alphasets[alphasets >= 0]
    negative_parameters = alphasets[alphasets < 0]

    bases_negative = tensorlib.tile(down_variation, negative_parameters.shape+(1,)*len(down_variation.shape))
    bases_negative = tensorlib.einsum('ij->ji', bases_negative)

    bases_positive = tensorlib.tile(up_variation, positive_parameters.shape+(1,)*len(up_variation.shape))
    bases_positive = tensorlib.einsum('ij->ji', bases_positive)

    expo_positive = tensorlib.tile(positive_parameters, up_variation.shape+(1,)) #was outer
    expo_negative = -tensorlib.tile(negative_parameters, down_variation.shape+(1,)) #was outer

    res_neg = tensorlib.power(bases_negative, expo_negative)
    res_pos = tensorlib.power(bases_positive, expo_positive)

    result = tensorlib.concatenate([res_neg, res_pos], axis=-1)
    return tensorlib.einsum('ijk->ikj', result)

def _kitchensink_code1(histogramssets, alphasets):
    def product(down, nom, up, alpha):
        delta_up = up/nom
        delta_down = down/nom
        if alpha > 0:
            delta =  delta_up**alpha
        else:
            delta =  delta_down**(-alpha)
        return nom*delta

    return _kitchensink_looper(histogramssets, alphasets, product)


# interpolation codes come from https://cds.cern.ch/record/1456844/files/CERN-OPEN-2012-016.pdf
def interpolator(interpcode, do_optimal=True):
    interpcodes = {0: _hfinterp_code0 if do_optimal else _kitchensink_code0,
                   1: _hfinterp_code1 if do_optimal else _kitchensink_code1}

    try:
        return interpcodes[interpcode]
    except KeyError:
        raise exceptions.InvalidInterpCode
