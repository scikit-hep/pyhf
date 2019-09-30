"""The probability density function module."""
from . import get_backend


class _SimpleDistributionMixin(object):
    """The mixin class for distributions."""

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
        The collection of values sampled from the probability density function.

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

    """

    def __init__(self, rate):
        """
        Args:
            rate (`tensor` or `float`): The mean of the Poisson distribution (the expected number of events)
        """
        tensorlib, _ = get_backend()
        self.rate = rate
        self._pdf = tensorlib.poisson_dist(rate)

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
    """

    def __init__(self, loc, scale):
        """
        Args:
            loc (`tensor` or `float`): The mean of the Normal distribution
            scale (`tensor` or `float`): The standard deviation of the Normal distribution
        """

        tensorlib, _ = get_backend()
        self.loc = loc
        self.scale = scale
        self._pdf = tensorlib.normal_dist(loc, scale)

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
    A probability density corresponding to the joint distribution of a batch of
    identically distributed random variables.

    Example:
        >>> import pyhf
        >>> import numpy.random as random
        >>> random.seed(0)
        >>> rates = pyhf.tensorlib.astensor([10.0, 10.0])
        >>> poissons = pyhf.probability.Poisson(rates)
        >>> independent = pyhf.probability.Independent(poissons)
        >>> independent.sample()
        array([10, 11])
    """

    def __init__(self, batched_pdf, batch_size=None):
        """
        Args:
            batched_pdf (`pyhf.probability` distribution): The batch of pdfs of the same type (e.g. Poisson)
            batch_size (`int`): The size of the batch
        """
        self.batch_size = batch_size
        self._pdf = batched_pdf

    def log_prob(self, value):
        r"""
        The log of the probability density function at the given value.
        As the distribution is a joint distribution of the same type, this is the
        sum of the log probabilities of each of the distributions the compose the joint.

        Example:
            >>> import pyhf
            >>> import numpy.random as random
            >>> random.seed(0)
            >>> rates = pyhf.tensorlib.astensor([10.0, 10.0])
            >>> poissons = pyhf.probability.Poisson(rates)
            >>> independent = pyhf.probability.Independent(poissons)
            >>> values = pyhf.tensorlib.astensor([8.0, 9.0])
            >>> independent.log_prob(values)
            -4.262483801927939
            >>> broadcast_value = pyhf.tensorlib.astensor([11.0])
            >>> independent.log_prob(broadcast_value)
            -4.347743645878765

        Args:
            value (`tensor` or `float`): The value at which to evaluate the distribution

        Returns:
            Tensor: The value of :math:`\log(f\left(x\middle|\theta\right))` for :math:`x=`:code:`value`

        """
        tensorlib, _ = get_backend()
        result = super(Independent, self).log_prob(value)
        result = tensorlib.sum(result, axis=-1)
        return result


class Simultaneous(object):
    """
    A probability density corresponding to the joint
    distribution of multiple non-identical component distributions

    Example:
        >>> import pyhf
        >>> import numpy.random as random
        >>> from pyhf.tensor.common import _TensorViewer
        >>> random.seed(0)
        >>> poissons = pyhf.probability.Poisson(pyhf.tensorlib.astensor([1.,100.]))
        >>> normals = pyhf.probability.Normal(pyhf.tensorlib.astensor([1.,100.]), pyhf.tensorlib.astensor([1.,2.]))
        >>> tv = _TensorViewer([[0,2],[1,3]])
        >>> sim = pyhf.probability.Simultaneous([poissons,normals], tv)
        >>> sim.sample((4,))
        array([[  2.        ,   1.3130677 , 101.        ,  98.29180852],
               [  1.        ,  -1.55298982,  97.        , 101.30723719],
               [  1.        ,   1.8644362 , 118.        ,  98.51566996],
               [  0.        ,   3.26975462,  99.        ,  97.09126865]])

    """

    def __init__(self, pdfobjs, tensorview, batch_size=None):
        """
        Construct a simultaneous pdf.

        Args:

            pdfobjs (`Distribution`): The constituent pdf objects
            tensorview (`_TensorViewer`): The :code:`_TensorViewer` defining the data composition
            batch_size (`int`): The size of the batch

        """
        self.tv = tensorview
        self._pdfobjs = pdfobjs
        self.batch_size = batch_size

    def __iter__(self):
        """
        Iterate over the constituent pdf objects

        Returns:
            pdfobj (`Distribution`): A constituent pdf object

        """
        for pdfobj in self._pdfobjs:
            yield pdfobj

    def __getitem__(self, index):
        """
        Access the constituent pdf object at the specified index

        Args:

            index (`int`): The index to access the constituent pdf object

        Returns:
            pdfobj (`Distribution`): A constituent pdf object

        """
        return self._pdfobjs[index]

    def expected_data(self):
        r"""
        The expectation value of the probability density function.

        Returns:
            Tensor: The expectation value of the distribution :math:`\mathrm{E}\left[f(\theta)\right]`

        """
        tostitch = [p.expected_data() for p in self]
        return self.tv.stitch(tostitch)

    def sample(self, sample_shape=()):
        r"""
        The collection of values sampled from the probability density function.

        Args:
            sample_shape (`tuple`): The shape of the sample to be returned

        Returns:
            Tensor: The values :math:`x \sim f(\theta)` where :math:`x` has shape :code:`sample_shape`

        """
        return self.tv.stitch([p.sample(sample_shape) for p in self])

    def log_prob(self, value):
        r"""
        The log of the probability density function at the given value.

        Args:
            value (`tensor`): The observed value

        Returns:
            Tensor: The value of :math:`\log(f\left(x\middle|\theta\right))` for :math:`x=`:code:`value`

        """
        constituent_data = self.tv.split(value)
        pdfvals = [p.log_prob(d) for p, d in zip(self, constituent_data)]
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
