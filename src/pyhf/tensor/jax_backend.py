from jax.config import config

config.update('jax_enable_x64', True)

import jax.numpy as np
from jax.scipy.special import gammaln
from jax.scipy.stats import norm, poisson
import numpy as onp
import logging

log = logging.getLogger(__name__)


class _BasicPoisson(object):
    def __init__(self, rate):
        self.rate = rate

    def sample(self, sample_shape):
        return poisson.osp_stats.poisson(self.rate).rvs(
            size=sample_shape + self.rate.shape
        )

    def log_prob(self, value):
        tensorlib = jax_backend()
        return tensorlib.poisson_logpdf(value, self.rate)


class _BasicNormal(object):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

    def sample(self, sample_shape):
        return norm.osp_stats.norm(self.loc, self.scale).rvs(
            size=sample_shape + self.loc.shape
        )

    def log_prob(self, value):
        tensorlib = jax_backend()
        return tensorlib.normal_logpdf(value, self.loc, self.scale)


class jax_backend(object):
    """JAX backend for pyhf"""

    __slots__ = ['name', 'precision', 'dtypemap', 'default_do_grad']

    def __init__(self, **kwargs):
        self.name = 'jax'
        self.precision = kwargs.get('precision', '64b')
        self.dtypemap = {
            'float': np.float64 if self.precision == '64b' else np.float32,
            'int': np.int64 if self.precision == '64b' else np.int32,
            'bool': np.bool_,
        }
        self.default_do_grad = True

    def _setup(self):
        """
        Run any global setups for the jax lib.
        """

    def clip(self, tensor_in, min_value, max_value):
        """
        Clips (limits) the tensor values to be within a specified min and max.

        Example:

            >>> import pyhf
            >>> pyhf.set_backend("jax")
            >>> a = pyhf.tensorlib.astensor([-2, -1, 0, 1, 2])
            >>> pyhf.tensorlib.clip(a, -1, 1)
            DeviceArray([-1., -1.,  0.,  1.,  1.], dtype=float64)

        Args:
            tensor_in (`tensor`): The input tensor object
            min_value (`scalar` or `tensor` or `None`): The minimum value to be cliped to
            max_value (`scalar` or `tensor` or `None`): The maximum value to be cliped to

        Returns:
            JAX ndarray: A clipped `tensor`
        """
        return np.clip(tensor_in, min_value, max_value)

    def tile(self, tensor_in, repeats):
        """
        Repeat tensor data along a specific dimension

        Example:

            >>> import pyhf
            >>> pyhf.set_backend("jax")
            >>> a = pyhf.tensorlib.astensor([[1.0], [2.0]])
            >>> pyhf.tensorlib.tile(a, (1, 2))
            DeviceArray([[1., 1.],
                         [2., 2.]], dtype=float64)

        Args:
            tensor_in (`Tensor`): The tensor to be repeated
            repeats (`Tensor`): The tuple of multipliers for each dimension

        Returns:
            JAX ndarray: The tensor with repeated axes
        """
        return np.tile(tensor_in, repeats)

    def conditional(self, predicate, true_callable, false_callable):
        """
        Runs a callable conditional on the boolean value of the evaulation of a predicate

        Example:

            >>> import pyhf
            >>> pyhf.set_backend("jax")
            >>> tensorlib = pyhf.tensorlib
            >>> a = tensorlib.astensor([4])
            >>> b = tensorlib.astensor([5])
            >>> tensorlib.conditional((a < b)[0], lambda: a + b, lambda: a - b)
            DeviceArray([9.], dtype=float64)

        Args:
            predicate (`scalar`): The logical condition that determines which callable to evaluate
            true_callable (`callable`): The callable that is evaluated when the :code:`predicate` evalutes to :code:`true`
            false_callable (`callable`): The callable that is evaluated when the :code:`predicate` evalutes to :code:`false`

        Returns:
            JAX ndarray: The output of the callable that was evaluated
        """
        return true_callable() if predicate else false_callable()

    def tolist(self, tensor_in):
        try:
            return onp.asarray(tensor_in).tolist()
        except AttributeError:
            if isinstance(tensor_in, list):
                return tensor_in
            raise

    def outer(self, tensor_in_1, tensor_in_2):
        return np.outer(tensor_in_1, tensor_in_2)

    def gather(self, tensor, indices):
        return tensor[indices]

    def boolean_mask(self, tensor, mask):
        return tensor[mask]

    def isfinite(self, tensor):
        return np.isfinite(tensor)

    def astensor(self, tensor_in, dtype='float'):
        """
        Convert to a JAX ndarray.

        Example:

            >>> import pyhf
            >>> pyhf.set_backend("jax")
            >>> tensor = pyhf.tensorlib.astensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            >>> tensor
            DeviceArray([[1., 2., 3.],
                         [4., 5., 6.]], dtype=float64)
            >>> type(tensor)
            <class 'jax.interpreters.xla.DeviceArray'>

        Args:
            tensor_in (Number or Tensor): Tensor object

        Returns:
            `jax.interpreters.xla.DeviceArray`: A multi-dimensional, fixed-size homogenous array.
        """
        try:
            dtype = self.dtypemap[dtype]
        except KeyError:
            log.error('Invalid dtype: dtype must be float, int, or bool.')
            raise

        return np.asarray(tensor_in, dtype=dtype)

    def sum(self, tensor_in, axis=None):
        return np.sum(tensor_in, axis=axis)

    def product(self, tensor_in, axis=None):
        return np.prod(tensor_in, axis=axis)

    def abs(self, tensor):
        return np.abs(tensor)

    def ones(self, shape):
        return np.ones(shape)

    def zeros(self, shape):
        return np.zeros(shape)

    def power(self, tensor_in_1, tensor_in_2):
        return np.power(tensor_in_1, tensor_in_2)

    def sqrt(self, tensor_in):
        return np.sqrt(tensor_in)

    def divide(self, tensor_in_1, tensor_in_2):
        return np.divide(tensor_in_1, tensor_in_2)

    def log(self, tensor_in):
        return np.log(tensor_in)

    def exp(self, tensor_in):
        return np.exp(tensor_in)

    def stack(self, sequence, axis=0):
        return np.stack(sequence, axis=axis)

    def where(self, mask, tensor_in_1, tensor_in_2):
        return np.where(mask, tensor_in_1, tensor_in_2)

    def concatenate(self, sequence, axis=0):
        """
        Join a sequence of arrays along an existing axis.

        Args:
            sequence: sequence of tensors
            axis: dimension along which to concatenate

        Returns:
            output: the concatenated tensor

        """
        return np.concatenate(sequence, axis=axis)

    def simple_broadcast(self, *args):
        """
        Broadcast a sequence of 1 dimensional arrays.

        Example:

            >>> import pyhf
            >>> pyhf.set_backend("jax")
            >>> pyhf.tensorlib.simple_broadcast(
            ...   pyhf.tensorlib.astensor([1]),
            ...   pyhf.tensorlib.astensor([2, 3, 4]),
            ...   pyhf.tensorlib.astensor([5, 6, 7]))
            [DeviceArray([1., 1., 1.], dtype=float64), DeviceArray([2., 3., 4.], dtype=float64), DeviceArray([5., 6., 7.], dtype=float64)]

        Args:
            args (Array of Tensors): Sequence of arrays

        Returns:
            list of Tensors: The sequence broadcast together.
        """
        return np.broadcast_arrays(*args)

    def shape(self, tensor):
        return tensor.shape

    def reshape(self, tensor, newshape):
        return np.reshape(tensor, newshape)

    def einsum(self, subscripts, *operands):
        """
        Evaluates the Einstein summation convention on the operands.

        Using the Einstein summation convention, many common multi-dimensional
        array operations can be represented in a simple fashion. This function
        provides a way to compute such summations. The best way to understand
        this function is to try the examples below, which show how many common
        NumPy functions can be implemented as calls to einsum.

        Args:
            subscripts: str, specifies the subscripts for summation
            operands: list of array_like, these are the tensors for the operation

        Returns:
            tensor: the calculation based on the Einstein summation convention
        """
        # return contract(subscripts,*operands)
        return np.einsum(subscripts, *operands)

    def poisson_logpdf(self, n, lam):
        n = np.asarray(n)
        lam = np.asarray(lam)
        return n * np.log(lam) - lam - gammaln(n + 1.0)

    def poisson(self, n, lam):
        r"""
        The continous approximation, using :math:`n! = \Gamma\left(n+1\right)`,
        to the probability mass function of the Poisson distribution evaluated
        at :code:`n` given the parameter :code:`lam`.

        Example:

            >>> import pyhf
            >>> pyhf.set_backend("jax")
            >>> pyhf.tensorlib.poisson(5., 6.)
            DeviceArray(0.16062314, dtype=float64)
            >>> values = pyhf.tensorlib.astensor([5., 9.])
            >>> rates = pyhf.tensorlib.astensor([6., 8.])
            >>> pyhf.tensorlib.poisson(values, rates)
            DeviceArray([0.16062314, 0.12407692], dtype=float64)

        Args:
            n (`tensor` or `float`): The value at which to evaluate the approximation to the Poisson distribution p.m.f.
                                  (the observed number of events)
            lam (`tensor` or `float`): The mean of the Poisson distribution p.m.f.
                                    (the expected number of events)

        Returns:
            JAX ndarray: Value of the continous approximation to Poisson(n|lam)
        """
        n = np.asarray(n)
        lam = np.asarray(lam)
        return np.exp(n * np.log(lam) - lam - gammaln(n + 1.0))

    def normal_logpdf(self, x, mu, sigma):
        # this is much faster than
        # norm.logpdf(x, loc=mu, scale=sigma)
        # https://codereview.stackexchange.com/questions/69718/fastest-computation-of-n-likelihoods-on-normal-distributions
        root2 = np.sqrt(2)
        root2pi = np.sqrt(2 * np.pi)
        prefactor = -np.log(sigma * root2pi)
        summand = -np.square(np.divide((x - mu), (root2 * sigma)))
        return prefactor + summand

    # def normal_logpdf(self, x, mu, sigma):
    #     return norm.logpdf(x, loc=mu, scale=sigma)

    def normal(self, x, mu, sigma):
        r"""
        The probability density function of the Normal distribution evaluated
        at :code:`x` given parameters of mean of :code:`mu` and standard deviation
        of :code:`sigma`.

        Example:

            >>> import pyhf
            >>> pyhf.set_backend("jax")
            >>> pyhf.tensorlib.normal(0.5, 0., 1.)
            DeviceArray(0.35206533, dtype=float64)
            >>> values = pyhf.tensorlib.astensor([0.5, 2.0])
            >>> means = pyhf.tensorlib.astensor([0., 2.3])
            >>> sigmas = pyhf.tensorlib.astensor([1., 0.8])
            >>> pyhf.tensorlib.normal(values, means, sigmas)
            DeviceArray([0.35206533, 0.46481887], dtype=float64)

        Args:
            x (`tensor` or `float`): The value at which to evaluate the Normal distribution p.d.f.
            mu (`tensor` or `float`): The mean of the Normal distribution
            sigma (`tensor` or `float`): The standard deviation of the Normal distribution

        Returns:
            JAX ndarray: Value of Normal(x|mu, sigma)
        """
        return norm.pdf(x, loc=mu, scale=sigma)

    def normal_cdf(self, x, mu=0, sigma=1):
        """
        The cumulative distribution function for the Normal distribution

        Example:

            >>> import pyhf
            >>> pyhf.set_backend("jax")
            >>> pyhf.tensorlib.normal_cdf(0.8)
            DeviceArray(0.7881446, dtype=float64)
            >>> values = pyhf.tensorlib.astensor([0.8, 2.0])
            >>> pyhf.tensorlib.normal_cdf(values)
            DeviceArray([0.7881446 , 0.97724987], dtype=float64)

        Args:
            x (`tensor` or `float`): The observed value of the random variable to evaluate the CDF for
            mu (`tensor` or `float`): The mean of the Normal distribution
            sigma (`tensor` or `float`): The standard deviation of the Normal distribution

        Returns:
            JAX ndarray: The CDF
        """
        return norm.cdf(x, loc=mu, scale=sigma)

    def poisson_dist(self, rate):
        r"""
        The Poisson distribution with rate parameter :code:`rate`.

        Example:
            >>> import pyhf
            >>> pyhf.set_backend("jax")
            >>> rates = pyhf.tensorlib.astensor([5, 8])
            >>> values = pyhf.tensorlib.astensor([4, 9])
            >>> poissons = pyhf.tensorlib.poisson_dist(rates)
            >>> poissons.log_prob(values)
            DeviceArray([-1.74030218, -2.0868536 ], dtype=float64)

        Args:
            rate (`tensor` or `float`): The mean of the Poisson distribution (the expected number of events)

        Returns:
            Poisson distribution: The Poisson distribution class
        """
        return _BasicPoisson(rate)

    def normal_dist(self, mu, sigma):
        r"""
        The Normal distribution with mean :code:`mu` and standard deviation :code:`sigma`.

        Example:
            >>> import pyhf
            >>> pyhf.set_backend("jax")
            >>> means = pyhf.tensorlib.astensor([5, 8])
            >>> stds = pyhf.tensorlib.astensor([1, 0.5])
            >>> values = pyhf.tensorlib.astensor([4, 9])
            >>> normals = pyhf.tensorlib.normal_dist(means, stds)
            >>> normals.log_prob(values)
            DeviceArray([-1.41893853, -2.22579135], dtype=float64)

        Args:
            mu (`tensor` or `float`): The mean of the Normal distribution
            sigma (`tensor` or `float`): The standard deviation of the Normal distribution

        Returns:
            Normal distribution: The Normal distribution class

        """
        return _BasicNormal(mu, sigma)
