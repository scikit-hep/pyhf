from jax.config import config

config.update('jax_enable_x64', True)

import jax.numpy as jnp
from jax.scipy.special import gammaln, xlogy
from jax.scipy import special
from jax.scipy.stats import norm
import numpy as np
import scipy.stats as osp_stats
import logging

log = logging.getLogger(__name__)

# v0.7.x backport hack
_old_jax_version = False
try:
    from jax import Array
except ImportError:
    # jax.Array added in jax v0.4.1
    _old_jax_version = True


class _BasicPoisson:
    def __init__(self, rate):
        self.rate = rate

    def sample(self, sample_shape):
        # TODO: Support other dtypes
        return jnp.asarray(
            osp_stats.poisson(self.rate).rvs(size=sample_shape + self.rate.shape),
            dtype=jnp.float64,
        )

    def log_prob(self, value):
        tensorlib = jax_backend()
        return tensorlib.poisson_logpdf(value, self.rate)


class _BasicNormal:
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

    def sample(self, sample_shape):
        # TODO: Support other dtypes
        return jnp.asarray(
            osp_stats.norm(self.loc, self.scale).rvs(
                size=sample_shape + self.loc.shape
            ),
            dtype=jnp.float64,
        )

    def log_prob(self, value):
        tensorlib = jax_backend()
        return tensorlib.normal_logpdf(value, self.loc, self.scale)


class jax_backend:
    """JAX backend for pyhf"""

    __slots__ = ['name', 'precision', 'dtypemap', 'default_do_grad']

    #: The array type for jax
    array_type = jnp.DeviceArray if _old_jax_version else Array

    #: The array content type for jax
    array_subtype = jnp.DeviceArray if _old_jax_version else Array

    def __init__(self, **kwargs):
        self.name = 'jax'
        self.precision = kwargs.get('precision', '64b')
        self.dtypemap = {
            'float': jnp.float64 if self.precision == '64b' else jnp.float32,
            'int': jnp.int64 if self.precision == '64b' else jnp.int32,
            'bool': jnp.bool_,
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
            Array([-1., -1.,  0.,  1.,  1.], dtype=float64)

        Args:
            tensor_in (:obj:`tensor`): The input tensor object
            min_value (:obj:`scalar` or :obj:`tensor` or :obj:`None`): The minimum value to be clipped to
            max_value (:obj:`scalar` or :obj:`tensor` or :obj:`None`): The maximum value to be clipped to

        Returns:
            JAX ndarray: A clipped `tensor`
        """
        return jnp.clip(tensor_in, min_value, max_value)

    def erf(self, tensor_in):
        """
        The error function of complex argument.

        Example:

            >>> import pyhf
            >>> pyhf.set_backend("jax")
            >>> a = pyhf.tensorlib.astensor([-2., -1., 0., 1., 2.])
            >>> pyhf.tensorlib.erf(a)
            Array([-0.99532227, -0.84270079,  0.        ,  0.84270079,  0.99532227],      dtype=float64)

        Args:
            tensor_in (:obj:`tensor`): The input tensor object

        Returns:
            JAX ndarray: The values of the error function at the given points.
        """
        return special.erf(tensor_in)

    def erfinv(self, tensor_in):
        """
        The inverse of the error function of complex argument.

        Example:

            >>> import pyhf
            >>> pyhf.set_backend("jax")
            >>> a = pyhf.tensorlib.astensor([-2., -1., 0., 1., 2.])
            >>> pyhf.tensorlib.erfinv(pyhf.tensorlib.erf(a))
            Array([-2., -1.,  0.,  1.,  2.], dtype=float64)

        Args:
            tensor_in (:obj:`tensor`): The input tensor object

        Returns:
            JAX ndarray: The values of the inverse of the error function at the given points.
        """
        return special.erfinv(tensor_in)

    def tile(self, tensor_in, repeats):
        """
        Repeat tensor data along a specific dimension

        Example:

            >>> import pyhf
            >>> pyhf.set_backend("jax")
            >>> a = pyhf.tensorlib.astensor([[1.0], [2.0]])
            >>> pyhf.tensorlib.tile(a, (1, 2))
            Array([[1., 1.],
                   [2., 2.]], dtype=float64)

        Args:
            tensor_in (:obj:`tensor`): The tensor to be repeated
            repeats (:obj:`tensor`): The tuple of multipliers for each dimension

        Returns:
            JAX ndarray: The tensor with repeated axes
        """
        return jnp.tile(tensor_in, repeats)

    def conditional(self, predicate, true_callable, false_callable):
        """
        Runs a callable conditional on the boolean value of the evaluation of a predicate

        Example:

            >>> import pyhf
            >>> pyhf.set_backend("jax")
            >>> tensorlib = pyhf.tensorlib
            >>> a = tensorlib.astensor([4])
            >>> b = tensorlib.astensor([5])
            >>> tensorlib.conditional((a < b)[0], lambda: a + b, lambda: a - b)
            Array([9.], dtype=float64)

        Args:
            predicate (:obj:`scalar`): The logical condition that determines which callable to evaluate
            true_callable (:obj:`callable`): The callable that is evaluated when the :code:`predicate` evaluates to :code:`true`
            false_callable (:obj:`callable`): The callable that is evaluated when the :code:`predicate` evaluates to :code:`false`

        Returns:
            JAX ndarray: The output of the callable that was evaluated
        """
        return true_callable() if predicate else false_callable()

    def tolist(self, tensor_in):
        try:
            return jnp.asarray(tensor_in).tolist()
        except (TypeError, ValueError):
            if isinstance(tensor_in, list):
                return tensor_in
            raise

    def outer(self, tensor_in_1, tensor_in_2):
        return jnp.outer(tensor_in_1, tensor_in_2)

    def gather(self, tensor, indices):
        return tensor[indices]

    def boolean_mask(self, tensor, mask):
        return tensor[mask]

    def isfinite(self, tensor):
        return jnp.isfinite(tensor)

    def astensor(self, tensor_in, dtype="float"):
        """
        Convert to a JAX ndarray.

        Example:

            >>> import pyhf
            >>> pyhf.set_backend("jax")
            >>> tensor = pyhf.tensorlib.astensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            >>> tensor
            Array([[1., 2., 3.],
                   [4., 5., 6.]], dtype=float64)
            >>> type(tensor) # doctest:+ELLIPSIS
            <class '...ArrayImpl'>

        Args:
            tensor_in (Number or Tensor): Tensor object

        Returns:
            `jaxlib.xla_extension.ArrayImpl`: A multi-dimensional, fixed-size homogeneous array.
        """
        # TODO: Remove doctest:+ELLIPSIS when JAX API stabilized
        try:
            dtype = self.dtypemap[dtype]
        except KeyError:
            log.error(
                'Invalid dtype: dtype must be float, int, or bool.', exc_info=True
            )
            raise

        return jnp.asarray(tensor_in, dtype=dtype)

    def sum(self, tensor_in, axis=None):
        return jnp.sum(tensor_in, axis=axis)

    def product(self, tensor_in, axis=None):
        return jnp.prod(tensor_in, axis=axis)

    def abs(self, tensor):
        return jnp.abs(tensor)

    def ones(self, shape, dtype="float"):
        try:
            dtype = self.dtypemap[dtype]
        except KeyError:
            log.error(
                f"Invalid dtype: dtype must be one of {list(self.dtypemap.keys())}.",
                exc_info=True,
            )
            raise

        return jnp.ones(shape, dtype=dtype)

    def zeros(self, shape, dtype="float"):
        try:
            dtype = self.dtypemap[dtype]
        except KeyError:
            log.error(
                f"Invalid dtype: dtype must be one of {list(self.dtypemap.keys())}.",
                exc_info=True,
            )
            raise

        return jnp.zeros(shape, dtype=dtype)

    def power(self, tensor_in_1, tensor_in_2):
        return jnp.power(tensor_in_1, tensor_in_2)

    def sqrt(self, tensor_in):
        return jnp.sqrt(tensor_in)

    def divide(self, tensor_in_1, tensor_in_2):
        return jnp.divide(tensor_in_1, tensor_in_2)

    def log(self, tensor_in):
        return jnp.log(tensor_in)

    def exp(self, tensor_in):
        return jnp.exp(tensor_in)

    def percentile(self, tensor_in, q, axis=None, interpolation="linear"):
        r"""
        Compute the :math:`q`-th percentile of the tensor along the specified axis.

        Example:

            >>> import pyhf
            >>> import jax.numpy as jnp
            >>> pyhf.set_backend("jax")
            >>> a = pyhf.tensorlib.astensor([[10, 7, 4], [3, 2, 1]])
            >>> pyhf.tensorlib.percentile(a, 50)
            Array(3.5, dtype=float64)
            >>> pyhf.tensorlib.percentile(a, 50, axis=1)
            Array([7., 2.], dtype=float64)

        Args:
            tensor_in (`tensor`): The tensor containing the data
            q (:obj:`float` or `tensor`): The :math:`q`-th percentile to compute
            axis (`number` or `tensor`): The dimensions along which to compute
            interpolation (:obj:`str`): The interpolation method to use when the
             desired percentile lies between two data points ``i < j``:

                - ``'linear'``: ``i + (j - i) * fraction``, where ``fraction`` is the
                  fractional part of the index surrounded by ``i`` and ``j``.

                - ``'lower'``: ``i``.

                - ``'higher'``: ``j``.

                - ``'midpoint'``: ``(i + j) / 2``.

                - ``'nearest'``: ``i`` or ``j``, whichever is nearest.

        Returns:
            JAX ndarray: The value of the :math:`q`-th percentile of the tensor along the specified axis.

        .. versionadded:: 0.7.0
        """
        return jnp.percentile(tensor_in, q, axis=axis, interpolation=interpolation)

    def stack(self, sequence, axis=0):
        return jnp.stack(sequence, axis=axis)

    def where(self, mask, tensor_in_1, tensor_in_2):
        return jnp.where(mask, tensor_in_1, tensor_in_2)

    def concatenate(self, sequence, axis=0):
        """
        Join a sequence of arrays along an existing axis.

        Args:
            sequence: sequence of tensors
            axis: dimension along which to concatenate

        Returns:
            output: the concatenated tensor

        """
        return jnp.concatenate(sequence, axis=axis)

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
            [Array([1., 1., 1.], dtype=float64), Array([2., 3., 4.], dtype=float64), Array([5., 6., 7.], dtype=float64)]

        Args:
            args (Array of Tensors): Sequence of arrays

        Returns:
            list of Tensors: The sequence broadcast together.
        """
        return jnp.broadcast_arrays(*args)

    def shape(self, tensor):
        return tensor.shape

    def reshape(self, tensor, newshape):
        return jnp.reshape(tensor, newshape)

    def ravel(self, tensor):
        """
        Return a flattened view of the tensor, not a copy.

        Example:

            >>> import pyhf
            >>> pyhf.set_backend("jax")
            >>> tensor = pyhf.tensorlib.astensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            >>> pyhf.tensorlib.ravel(tensor)
            Array([1., 2., 3., 4., 5., 6.], dtype=float64)

        Args:
            tensor (Tensor): Tensor object

        Returns:
            `jaxlib.xla_extension.Array`: A flattened array.
        """
        return jnp.ravel(tensor)

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
        return jnp.einsum(subscripts, *operands)

    def poisson_logpdf(self, n, lam):
        n = jnp.asarray(n)
        lam = jnp.asarray(lam)
        return xlogy(n, lam) - lam - gammaln(n + 1.0)

    def poisson(self, n, lam):
        r"""
        The continuous approximation, using :math:`n! = \Gamma\left(n+1\right)`,
        to the probability mass function of the Poisson distribution evaluated
        at :code:`n` given the parameter :code:`lam`.

        .. note::

            Though the p.m.f of the Poisson distribution is not defined for
            :math:`\lambda = 0`, the limit as :math:`\lambda \to 0` is still
            defined, which gives a degenerate p.m.f. of

            .. math::

                \lim_{\lambda \to 0} \,\mathrm{Pois}(n | \lambda) =
                \left\{\begin{array}{ll}
                1, & n = 0,\\
                0, & n > 0
                \end{array}\right.

        Example:

            >>> import pyhf
            >>> pyhf.set_backend("jax")
            >>> pyhf.tensorlib.poisson(5., 6.)
            Array(0.16062314, dtype=float64, weak_type=True)
            >>> values = pyhf.tensorlib.astensor([5., 9.])
            >>> rates = pyhf.tensorlib.astensor([6., 8.])
            >>> pyhf.tensorlib.poisson(values, rates)
            Array([0.16062314, 0.12407692], dtype=float64)

        Args:
            n (:obj:`tensor` or :obj:`float`): The value at which to evaluate the approximation to the Poisson distribution p.m.f.
                                  (the observed number of events)
            lam (:obj:`tensor` or :obj:`float`): The mean of the Poisson distribution p.m.f.
                                    (the expected number of events)

        Returns:
            JAX ndarray: Value of the continuous approximation to Poisson(n|lam)
        """
        n = jnp.asarray(n)
        lam = jnp.asarray(lam)
        return jnp.exp(xlogy(n, lam) - lam - gammaln(n + 1.0))

    def normal_logpdf(self, x, mu, sigma):
        # this is much faster than
        # norm.logpdf(x, loc=mu, scale=sigma)
        # https://codereview.stackexchange.com/questions/69718/fastest-computation-of-n-likelihoods-on-normal-distributions
        root2 = jnp.sqrt(2)
        root2pi = jnp.sqrt(2 * jnp.pi)
        prefactor = -jnp.log(sigma * root2pi)
        summand = -jnp.square(jnp.divide((x - mu), (root2 * sigma)))
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
            Array(0.35206533, dtype=float64, weak_type=True)
            >>> values = pyhf.tensorlib.astensor([0.5, 2.0])
            >>> means = pyhf.tensorlib.astensor([0., 2.3])
            >>> sigmas = pyhf.tensorlib.astensor([1., 0.8])
            >>> pyhf.tensorlib.normal(values, means, sigmas)
            Array([0.35206533, 0.46481887], dtype=float64)

        Args:
            x (:obj:`tensor` or :obj:`float`): The value at which to evaluate the Normal distribution p.d.f.
            mu (:obj:`tensor` or :obj:`float`): The mean of the Normal distribution
            sigma (:obj:`tensor` or :obj:`float`): The standard deviation of the Normal distribution

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
            Array(0.7881446, dtype=float64)
            >>> values = pyhf.tensorlib.astensor([0.8, 2.0])
            >>> pyhf.tensorlib.normal_cdf(values)
            Array([0.7881446 , 0.97724987], dtype=float64)

        Args:
            x (:obj:`tensor` or :obj:`float`): The observed value of the random variable to evaluate the CDF for
            mu (:obj:`tensor` or :obj:`float`): The mean of the Normal distribution
            sigma (:obj:`tensor` or :obj:`float`): The standard deviation of the Normal distribution

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
            Array([-1.74030218, -2.0868536 ], dtype=float64)

        Args:
            rate (:obj:`tensor` or :obj:`float`): The mean of the Poisson distribution (the expected number of events)

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
            Array([-1.41893853, -2.22579135], dtype=float64)

        Args:
            mu (:obj:`tensor` or :obj:`float`): The mean of the Normal distribution
            sigma (:obj:`tensor` or :obj:`float`): The standard deviation of the Normal distribution

        Returns:
            Normal distribution: The Normal distribution class

        """
        return _BasicNormal(mu, sigma)

    def to_numpy(self, tensor_in):
        """
        Convert the JAX tensor to a :class:`numpy.ndarray`.

        Example:
            >>> import pyhf
            >>> pyhf.set_backend("jax")
            >>> tensor = pyhf.tensorlib.astensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            >>> tensor
            Array([[1., 2., 3.],
                   [4., 5., 6.]], dtype=float64)
            >>> numpy_ndarray = pyhf.tensorlib.to_numpy(tensor)
            >>> numpy_ndarray
            array([[1., 2., 3.],
                   [4., 5., 6.]])
            >>> type(numpy_ndarray)
            <class 'numpy.ndarray'>

        Args:
            tensor_in (:obj:`tensor`): The input tensor object.

        Returns:
            :class:`numpy.ndarray`: The tensor converted to a NumPy ``ndarray``.

        """
        return np.asarray(tensor_in, dtype=tensor_in.dtype)

    def transpose(self, tensor_in):
        """
        Transpose the tensor.

        Example:
            >>> import pyhf
            >>> pyhf.set_backend("jax")
            >>> tensor = pyhf.tensorlib.astensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            >>> tensor
            Array([[1., 2., 3.],
                   [4., 5., 6.]], dtype=float64)
            >>> pyhf.tensorlib.transpose(tensor)
            Array([[1., 4.],
                   [2., 5.],
                   [3., 6.]], dtype=float64)

        Args:
            tensor_in (:obj:`tensor`): The input tensor object.

        Returns:
            JAX ndarray: The transpose of the input tensor.

        .. versionadded:: 0.7.0
        """
        return tensor_in.transpose()
