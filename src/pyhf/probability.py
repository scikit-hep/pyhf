from . import get_backend
from .tensor.common import TensorViewer


class ForwardMixin(object):
    def log_prob(self, value):
        return self._pdf.log_prob(value)

    def expected_data(self):
        return self._pdf.expected_data()

    def sample(self, sample_shape=()):
        return self._pdf.sample(sample_shape)


class Poisson(ForwardMixin):
    def __init__(self, rate):
        tensorlib, _ = get_backend()
        self.lam = tensorlib.astensor(rate)
        self._pdf = tensorlib.poisson_pdfcls(rate)

    def expected_data(self):
        return self.lam


class Normal(ForwardMixin):
    def __init__(self, loc, scale):
        tensorlib, _ = get_backend()
        self.mu = tensorlib.astensor(loc)
        self.sigma = tensorlib.astensor(scale)
        self._pdf = tensorlib.normal_pdfcls(loc, scale)

    def expected_data(self):
        return self.mu


class Independent(object):
    '''
    A probability density corresponding to the joint
    likelihood of a batch of identically distributed random
    numbers.
    '''

    def __init__(self, batched_pdf, batch_size=None):
        self.batch_size = batch_size
        self._pdf = batched_pdf

    def expected_data(self):
        return self._pdf.expected_data()

    def sample(self, sample_shape=()):
        return self._pdf.sample(sample_shape)

    def log_prob(self, value):
        tensorlib, _ = get_backend()
        result = self._pdf.log_prob(value)
        result = tensorlib.sum(result, axis=-1)
        return result


class Simultaneous(object):
    def __init__(self, pdfobjs, indices):
        self.tv = TensorViewer(indices)
        self.pdfobjs = pdfobjs

    def log_prob(self, data):
        constituent_data = self.tv.split(data)
        pdfvals = [p.log_prob(d) for p, d in zip(self.pdfobjs, constituent_data)]
        return joint_logpdf(pdfvals)

    def expected_data(self):
        tostitch = [p.expected_data() for p in self.pdfobjs]
        return self.tv.stitch(tostitch)

    def sample(self, sample_shape=()):
        return self.tv.stitch([p.sample(sample_shape) for p in self.pdfobjs])


def joint_logpdf(terms):
    tensorlib, _ = get_backend()
    terms = tensorlib.stack(terms)
    result = tensorlib.sum(terms, axis=0)
    return result
