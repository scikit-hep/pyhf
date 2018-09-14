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

    # these three variables can be pre-computed at PDF time
    allset_allhisto_nom = histogramssets[:,:,1]
    allset_allhisto_deltas_up = histogramssets[:,:,2] - histogramssets[:,:,1]
    allset_allhisto_deltas_dn = histogramssets[:,:,1] - histogramssets[:,:,0]

    where_alphasets_positive = tensorlib.where(alphasets > 0, tensorlib.ones(alphasets.shape), tensorlib.zeros(alphasets.shape))

    # s: set under consideration (i.e. the modifier)
    # a: alpha variation
    # h: histogram affected by modifier
    # b: bin of histogram
    allsets_allhistos_alphas_times_deltas_up = tensorlib.einsum('sa,shb->shab',alphasets,allset_allhisto_deltas_up)
    allsets_allhistos_alphas_times_deltas_dn = tensorlib.einsum('sa,shb->shab',alphasets,allset_allhisto_deltas_dn)
    allsets_allhistos_masks = tensorlib.einsum('sa,shb->shab', where_alphasets_positive, tensorlib.ones(allset_allhisto_deltas_dn.shape))

    allsets_allhistos_deltas = tensorlib.where(allsets_allhistos_masks,allsets_allhistos_alphas_times_deltas_up, allsets_allhistos_alphas_times_deltas_dn)
    allsets_allhistos_noms_repeated = tensorlib.einsum('sa,shb->shab',tensorlib.ones(alphasets.shape),allset_allhisto_nom)
    set_results = allsets_allhistos_deltas + allsets_allhistos_noms_repeated
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

    # these three variables can be pre-computed at PDF time
    allset_allhisto_nom = histogramssets[:,:,1]
    allset_allhisto_deltas_up = tensorlib.divide(histogramssets[:,:,2], allset_allhisto_nom)
    allset_allhisto_deltas_dn = tensorlib.divide(histogramssets[:,:,0], allset_allhisto_nom)

    allsets_allhistos_masks = tensorlib.where(alphasets > 0, tensorlib.ones(alphasets.shape), tensorlib.zeros(alphasets.shape))

    # s: set under consideration (i.e. the modifier)
    # a: alpha variation
    # h: histogram affected by modifier
    # b: bin of histogram
    bases_up = tensorlib.einsum('sa,shb->shab', tensorlib.ones(alphasets.shape), allset_allhisto_deltas_up)
    bases_dn = tensorlib.einsum('sa,shb->shab', tensorlib.ones(alphasets.shape), allset_allhisto_deltas_dn)
    exponents = tensorlib.einsum('sa,shb->shab', tensorlib.abs(alphasets), tensorlib.ones(allset_allhisto_deltas_up.shape))
    masks = tensorlib.einsum('sa,shb->shab', allsets_allhistos_masks, tensorlib.ones(allset_allhisto_deltas_up.shape))
    allsets_allhistos_noms_repeated = tensorlib.einsum('sa,shb->shab', tensorlib.ones(alphasets.shape), allset_allhisto_nom)

    bases = tensorlib.where(masks, bases_up, bases_dn)
    allsets_allhistos_deltas = tensorlib.power(bases, exponents)
    set_results = allsets_allhistos_deltas * allsets_allhistos_noms_repeated
    return set_results

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
