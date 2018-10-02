import logging
log = logging.getLogger(__name__)

from . import get_backend
from . import exceptions

class Interpolator(object):
    def __init__(self, histogramssets):
        self.tensorlib = None
        self.shape = None
        self.histogramssets = histogramssets
        self._precompute(shape=(histogramssets.shape[0], 1))

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

class _hfinterpolator_code0(Interpolator):
    def _precompute(self, shape):
        tensorlib, _ = get_backend()

        # did things change that we need to recompute?
        tensor_type_change = tensorlib != self.tensorlib
        shape_change = shape != self.shape

        if tensor_type_change:
            # nb: self.histogramssets will never change
            histogramssets = tensorlib.astensor(self.histogramssets)
            self.allset_allhisto_deltas_up = histogramssets[:,:,2] - histogramssets[:,:,1]
            self.allset_allhisto_deltas_dn = histogramssets[:,:,1] - histogramssets[:,:,0]
            self.broadcast_helper = tensorlib.ones(self.allset_allhisto_deltas_up.shape)

        if shape_change or tensor_type_change:
            self.mask_on  = tensorlib.ones(shape)
            self.mask_off = tensorlib.zeros(shape)

        # update after recomputation if needed
        self.tensorlib = tensorlib
        self.shape = shape

        return True

    def __call__(self, alphasets):
        tensorlib, _ = get_backend()
        self._precompute(alphasets.shape)
        where_alphasets_positive = tensorlib.where(alphasets > 0, self.mask_on, self.mask_off)

        # s: set under consideration (i.e. the modifier)
        # a: alpha variation
        # h: histogram affected by modifier
        # b: bin of histogram
        allsets_allhistos_alphas_times_deltas_up = tensorlib.einsum('sa,shb->shab',alphasets,self.allset_allhisto_deltas_up)
        allsets_allhistos_alphas_times_deltas_dn = tensorlib.einsum('sa,shb->shab',alphasets,self.allset_allhisto_deltas_dn)

        allsets_allhistos_masks = tensorlib.einsum('sa,shb->shab', where_alphasets_positive, self.broadcast_helper)

        return tensorlib.where(allsets_allhistos_masks,allsets_allhistos_alphas_times_deltas_up, allsets_allhistos_alphas_times_deltas_dn)

def _hfinterp_code0(histogramssets, alphasets):
    tensorlib, _ = get_backend()
    interpolator = _hfinterpolator_code0(histogramssets)
    return interpolator(alphasets)

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
    interpolator = _hfinterpolator_code1(histogramssets)
    return interpolator(alphasets)

class _hfinterpolator_code1(Interpolator):
    def _precompute(self, shape):
        tensorlib, _ = get_backend()

        # did things change that we need to recompute?
        tensor_type_change = tensorlib != self.tensorlib
        shape_change = shape != self.shape

        if tensor_type_change:
            # nb: self.histogramssets will never change
            histogramssets = tensorlib.astensor(self.histogramssets)
            self.allset_allhisto_deltas_up = tensorlib.divide(histogramssets[:,:,2], histogramssets[:,:,1])
            self.allset_allhisto_deltas_dn = tensorlib.divide(histogramssets[:,:,0], histogramssets[:,:,1])
            self.broadcast_helper = tensorlib.ones(self.allset_allhisto_deltas_up.shape)

        if shape_change or tensor_type_change:
            self.bases_up = tensorlib.einsum('sa,shb->shab', tensorlib.ones(shape), self.allset_allhisto_deltas_up)
            self.bases_dn = tensorlib.einsum('sa,shb->shab', tensorlib.ones(shape), self.allset_allhisto_deltas_dn)
            self.mask_on  = tensorlib.ones(shape)
            self.mask_off = tensorlib.zeros(shape)

        # update after recomputation if needed
        self.tensorlib = tensorlib
        self.shape = shape

        return True

    def __call__(self, alphasets):
        tensorlib, _ = get_backend()
        self._precompute(alphasets.shape)
        allsets_allhistos_masks = tensorlib.where(alphasets > 0, self.mask_on,self.mask_off)

        # s: set under consideration (i.e. the modifier)
        # a: alpha variation
        # h: histogram affected by modifier
        # b: bin of histogram
        exponents = tensorlib.einsum('sa,shb->shab', tensorlib.abs(alphasets), self.broadcast_helper)
        masks = tensorlib.einsum('sa,shb->shab', allsets_allhistos_masks, self.broadcast_helper)

        bases = tensorlib.where(masks, self.bases_up, self.bases_dn)
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
