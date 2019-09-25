from . import get_backend


class Poisson(object):
    def __init__(self, rate):
        self.rate = rate

    def log_prob(self, value):
        tensorlib, _ = get_backend()
        return tensorlib.poisson_logpdf(value, self.rate)


class Normal(object):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

    def log_prob(self, value):
        tensorlib, _ = get_backend()
        return tensorlib.normal_logpdf(value, self.loc, self.scale)


class Independent(object):
    """
    A probability density corresponding to the joint
    distribution of a batch of identically distributed random
    numbers.
    """

    def __init__(self, batched_pdf, batch_size=None):
        self.batch_size = batch_size
        self._pdf = batched_pdf

    def log_prob(self, value):
        tensorlib, _ = get_backend()
        _log_prob = self._pdf.log_prob(value)
        return tensorlib.sum(_log_prob, axis=-1)


class Simultaneous(object):
    def __init__(self, pdfobjs, tensorview, batch_size):
        self.tv = tensorview
        self.pdfobjs = pdfobjs
        self.batch_size = batch_size

    def log_prob(self, data):
        constituent_data = self.tv.split(data)
        pdfvals = [p.log_prob(d) for p, d in zip(self.pdfobjs, constituent_data)]
        return joint_logpdf(pdfvals, batch_size=self.batch_size)


def joint_logpdf(terms, batch_size=None):
    tensorlib, _ = get_backend()
    if len(terms) == 1:
        return terms[0]
    if len(terms) == 2 and batch_size is None:
        return terms[0] + terms[1]
    terms = tensorlib.stack(terms)
    return tensorlib.sum(terms, axis=0)
