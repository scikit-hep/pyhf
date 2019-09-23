from . import get_backend


class Poisson(object):
    def __init__(self, rate):
        tensorlib, _ = get_backend()
        self.lam = tensorlib.astensor(rate)

    def log_prob(self, value):
        tensorlib, _ = get_backend()
        n = tensorlib.astensor(value)
        return tensorlib.poisson_logpdf(n, self.lam)


class Normal(object):
    def __init__(self, loc, scale):
        tensorlib, _ = get_backend()
        self.mu = tensorlib.astensor(loc)
        self.sigma = tensorlib.astensor(scale)

    def log_prob(self, value):
        tensorlib, _ = get_backend()
        return tensorlib.normal_logpdf(value, self.mu, self.sigma)


class Independent(object):
    '''
    A probability density corresponding to the joint
    distribution of a batch of identically distributed random
    numbers.
    '''

    def __init__(self, batched_pdf, batch_size=None):
        self.batch_size = batch_size
        self._pdf = batched_pdf

    def log_prob(self, value):
        tensorlib, _ = get_backend()
        _log_prob = self._pdf.log_prob(value)
        return tensorlib.sum(_log_prob, axis=-1)


def joint_logpdf(terms, batch_size):
    tensorlib, _ = get_backend()
    if len(terms) == 1:
        return terms[0]
    if len(terms) == 2 and batch_size is None:
        return terms[0] + terms[1]
    terms = tensorlib.stack(terms)
    return tensorlib.sum(terms, axis=0)
