import logging
log = logging.getLogger(__name__)

from . import get_backend
from . import exceptions

def _slow_hfinterp_looper(histogramssets, alphasets, func):
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

def _hfinterp_code0(histogramssets, alphasets):
    tensorlib, _ = get_backend()

    # these two variables can be pre-computed at PDF time
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

    return tensorlib.where(allsets_allhistos_masks,allsets_allhistos_alphas_times_deltas_up, allsets_allhistos_alphas_times_deltas_dn)

def _slow_hfinterp_code0(histogramssets, alphasets):
    def summand(down, nom, up, alpha):
        delta_up = up - nom
        delta_down = nom - down
        if alpha > 0:
            delta =  delta_up*alpha
        else:
            delta =  delta_down*alpha
        return delta

    return _slow_hfinterp_looper(histogramssets, alphasets, summand)

def _hfinterp_code1(histogramssets, alphasets):
    tensorlib, _ = get_backend()

    # these two variables can be pre-computed at PDF time
    allset_allhisto_deltas_up = tensorlib.divide(histogramssets[:,:,2], histogramssets[:,:,1])
    allset_allhisto_deltas_dn = tensorlib.divide(histogramssets[:,:,0], histogramssets[:,:,1])

    allsets_allhistos_masks = tensorlib.where(alphasets > 0, tensorlib.ones(alphasets.shape), tensorlib.zeros(alphasets.shape))

    # s: set under consideration (i.e. the modifier)
    # a: alpha variation
    # h: histogram affected by modifier
    # b: bin of histogram
    bases_up = tensorlib.einsum('sa,shb->shab', tensorlib.ones(alphasets.shape), allset_allhisto_deltas_up)
    bases_dn = tensorlib.einsum('sa,shb->shab', tensorlib.ones(alphasets.shape), allset_allhisto_deltas_dn)
    exponents = tensorlib.einsum('sa,shb->shab', tensorlib.abs(alphasets), tensorlib.ones(allset_allhisto_deltas_up.shape))
    masks = tensorlib.einsum('sa,shb->shab', allsets_allhistos_masks, tensorlib.ones(allset_allhisto_deltas_up.shape))

    bases = tensorlib.where(masks, bases_up, bases_dn)
    return tensorlib.power(bases, exponents)

def _slow_hfinterp_code1(histogramssets, alphasets):
    def product(down, nom, up, alpha):
        delta_up = up/nom
        delta_down = down/nom
        if alpha > 0:
            delta =  delta_up**alpha
        else:
            delta =  delta_down**(-alpha)
        return delta

    return _slow_hfinterp_looper(histogramssets, alphasets, product)


# interpolation codes come from https://cds.cern.ch/record/1456844/files/CERN-OPEN-2012-016.pdf
def interpolator(interpcode, do_tensorized_calc=True):
    interpcodes = {0: _hfinterp_code0 if do_tensorized_calc else _slow_hfinterp_code0,
                   1: _hfinterp_code1 if do_tensorized_calc else _slow_hfinterp_code1}

    try:
        return interpcodes[interpcode]
    except KeyError:
        raise exceptions.InvalidInterpCode
