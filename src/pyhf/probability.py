from . import get_backend


class _SimpleDistributionMixin(object):
    """
    The mixin class for distributions
    """

    def log_prob(self, value):
        r"""
        The log of the probability density function at the given value.

        Args:
            value (`tensor` or `float`): The value at which to evaluate the distribution

        Returns:
            Tensor: The value of :math:`\log(f\left(x\middle|\theta\right))` for :math:`x=`:code:`value`
        """
        return self._pdf.log_prob(value)

    def expected_data(self):
        r"""
        The expectation value of the probability density function.

        Returns:
            Tensor: The expectation value of the distribution :math:`\mathrm{E}\left[f(\theta)\right]`
        """
        return self._pdf.expected_data()

    def sample(self, sample_shape=()):
        r"""
        The collection of values sampled from the probability density function

        Args:
            sample_shape (`tuple`): The shape of the sample to be returned

        Returns:
            Tensor: The values :math:`x \sim f(\theta)` where :math:`x` has shape :code:`sample_shape`
        """
        return self._pdf.sample(sample_shape)


class Poisson(_SimpleDistributionMixin):
    r"""
    The Poisson distribution with rate parameter :code:`rate`.

    Example:
        >>> import pyhf
        >>> rates = pyhf.tensorlib.astensor([5, 8])
        >>> pyhf.probability.Poisson(rates)
        <pyhf.probability.Poisson object at 0x...>

    Args:
        rate (`tensor` or `float`): The mean of the Poisson distribution (the expected number of events)
    """

    def __init__(self, rate):
        tensorlib, _ = get_backend()
        self.rate = rate
        self._pdf = tensorlib.poisson_pdfcls(rate)

    def expected_data(self):
        r"""
        The expectation value of the Poisson distribution.

        Example:
            >>> import pyhf
            >>> rates = pyhf.tensorlib.astensor([5, 8])
            >>> poissons = pyhf.probability.Poisson(rates)
            >>> poissons.expected_data()
            array([5., 8.])

        Returns:
            Tensor: The mean of the Poisson distribution (which is the :code:`rate`)
        """
        return self.rate


class Normal(_SimpleDistributionMixin):
    r"""
    The Normal distribution with mean :code:`loc` and standard deviation :code:`scale`.

    Example:
        >>> import pyhf
        >>> means = pyhf.tensorlib.astensor([5, 8])
        >>> stds = pyhf.tensorlib.astensor([1, 0.5])
        >>> pyhf.probability.Normal(means, stds)
        <pyhf.probability.Normal object at 0x...>

    Args:
        loc (`tensor` or `float`): The mean of the Normal distribution
        scale (`tensor` or `float`): The standard deviation of the Normal distribution
    """

    def __init__(self, loc, scale):
        tensorlib, _ = get_backend()
        self.loc = loc
        self.scale = scale
        self._pdf = tensorlib.normal_pdfcls(loc, scale)

    def expected_data(self):
        r"""
        The expectation value of the Normal distribution.

        Example:
            >>> import pyhf
            >>> means = pyhf.tensorlib.astensor([5, 8])
            >>> stds = pyhf.tensorlib.astensor([1, 0.5])
            >>> normals = pyhf.probability.Normal(means, stds)
            >>> normals.expected_data()
            array([5., 8.])

        Returns:
            Tensor: The mean of the Normal distribution (which is the :code:`loc`)
        """
        return self.loc


class Independent(_SimpleDistributionMixin):
    """
    A probability density corresponding to the joint
    likelihood of a batch of identically distributed random
    numbers.
    """

    def __init__(self, batched_pdf, batch_size=None):
        self.batch_size = batch_size
        self._pdf = batched_pdf

    def log_prob(self, value):
        tensorlib, _ = get_backend()
        result = super(Independent, self).log_prob(value)
        result = tensorlib.sum(result, axis=-1)
        return result


class Simultaneous(object):
    """
    A probability density corresponding to the joint
    likelihood multiple non-identical component distributions
    """

    def __init__(self, pdfobjs, tensorview, batch_size):
        self.tv = tensorview
        self.pdfobjs = pdfobjs
        self.batch_size = batch_size

    def expected_data(self):
        tostitch = [p.expected_data() for p in self.pdfobjs]
        return self.tv.stitch(tostitch)

    def sample(self, sample_shape=()):
        return self.tv.stitch([p.sample(sample_shape) for p in self.pdfobjs])

    def log_prob(self, data):
        constituent_data = self.tv.split(data)
        pdfvals = [p.log_prob(d) for p, d in zip(self.pdfobjs, constituent_data)]
        return Simultaneous._joint_logpdf(pdfvals, batch_size=self.batch_size)

    @staticmethod
    def _joint_logpdf(terms, batch_size=None):
        tensorlib, _ = get_backend()
        if len(terms) == 1:
            return terms[0]
        if len(terms) == 2 and batch_size is None:
            return terms[0] + terms[1]
        terms = tensorlib.stack(terms)
        return tensorlib.sum(terms, axis=0)
