from .. import get_backend
import numpy as np


class Simultaneous(object):
    def __init__(self, factors, projections):
        self.factors = factors
        self.projections = projections
        self.batch_shape = (sum([x.batch_shape[0] for x in self.factors]),)

    def sample(self, sample_shape=None):
        tensorlib, _ = get_backend()
        pieces = []
        for fac in self.factors:
            pieces.append(
                fac.sample(sample_shape if sample_shape is not None else (1,))
            )
        result = tensorlib.concatenate(pieces, axis=-1)
        return result if sample_shape is not None else result[0]

    def log_prob(self, value):
        tensorlib, _ = get_backend()
        value = tensorlib.astensor(value, dtype='float')
        log_sum = []
        flat = False

        if len(tensorlib.shape(value)) == 1:
            flat = True
            value = tensorlib.reshape(value, (1, -1))
        for fac_index, fac in enumerate(self.factors):
            projected = self.project(value, fac_index)
            projected = fac.log_prob(projected)
            log_sum.append(tensorlib.reshape(projected, (value.shape[0], -1)))
        result = tensorlib.reshape(
            tensorlib.sum(tensorlib.concatenate(log_sum, axis=-1), axis=-1),
            (value.shape[0],),
        )
        if flat:
            return result[0]
        return result

    def project(self, data, factor_index):
        tensorlib, _ = get_backend()
        if len(tensorlib.shape(data)) == 1:
            return tensorlib.gather(data, self.projections[factor_index])
        mask = np.zeros(self.batch_shape[0])
        mask[self.projections[factor_index]] = 1
        mask = mask.reshape(1, -1)
        mask = tensorlib.astensor(mask, dtype='bool')
        mask = [mask] * data.shape[0]
        mask = tensorlib.concatenate(mask)
        return tensorlib.reshape(
            tensorlib.boolean_mask(data, mask), (data.shape[0], -1)
        )
