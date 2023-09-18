"""NumPy Tensor Library Module."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Generic, Mapping, Sequence, TypeVar, Union

import numpy as np

# Needed while numpy lower bound is older than v1.21.0
if TYPE_CHECKING:
    from numpy.typing import ArrayLike, DTypeLike, NBitBase, NDArray
else:
    NBitBase = "NBitBase"

from scipy import special
from scipy.special import gammaln, xlogy
from scipy.stats import norm, poisson

from pyhf.typing import Literal, Shape

T = TypeVar("T", bound=NBitBase)

Tensor = Union["NDArray[np.number[T]]", "NDArray[np.bool_]"]
FloatIntOrBool = Literal["float", "int", "bool"]
log = logging.getLogger(__name__)


class _BasicPoisson:
    def __init__(self, rate: Tensor[T]):
        self.rate = rate

    def sample(self, sample_shape: Shape) -> ArrayLike:
        return poisson(self.rate).rvs(size=sample_shape + self.rate.shape)  # type: ignore[no-any-return]

    def log_prob(self, value: NDArray[np.number[T]]) -> ArrayLike:
        tensorlib: numpy_backend[T] = numpy_backend()
        return tensorlib.poisson_logpdf(value, self.rate)


class _BasicNormal:
    def __init__(self, loc: Tensor[T], scale: Tensor[T]):
        self.loc = loc
        self.scale = scale

    def sample(self, sample_shape: Shape) -> ArrayLike:
        return norm(self.loc, self.scale).rvs(size=sample_shape + self.loc.shape)  # type: ignore[no-any-return]

    def log_prob(self, value: NDArray[np.number[T]]) -> ArrayLike:
        tensorlib: numpy_backend[T] = numpy_backend()
        return tensorlib.normal_logpdf(value, self.loc, self.scale)


class numpy_backend(Generic[T]):
    """NumPy backend for pyhf"""

    __slots__ = ['name', 'precision', 'dtypemap', 'default_do_grad']

    #: The array type for numpy
    array_type = np.ndarray

    #: The array content type for numpy
    array_subtype = np.number

    def __init__(self, **kwargs: str):
        self.name = "numpy"
        self.precision = kwargs.get("precision", "64b")
        self.dtypemap: Mapping[
            FloatIntOrBool,
            DTypeLike,  # Type[np.floating[T]] | Type[np.integer[T]] | Type[np.bool_],
        ] = {
            'float': np.float64 if self.precision == '64b' else np.float32,
            'int': np.int64 if self.precision == '64b' else np.int32,
            'bool': np.bool_,
        }
        self.default_do_grad: bool = False

    def _setup(self) -> None:
        """
        Run any global setups for the numpy lib.
        """

    def clip(
        self,
        tensor_in: Tensor[T],
        min_value: np.integer[T] | np.floating[T],
        max_value: np.integer[T] | np.floating[T],
    ) -> ArrayLike:
        """
        Clips (limits) the tensor values to be within a specified min and max.

        Example:

            >>> import pyhf
            >>> pyhf.set_backend("numpy")
            >>> a = pyhf.tensorlib.astensor([-2, -1, 0, 1, 2])
            >>> pyhf.tensorlib.clip(a, -1, 1)
            array([-1., -1.,  0.,  1.,  1.])

        Args:
            tensor_in (:obj:`tensor`): The input tensor object
            min_value (:obj:`scalar` or :obj:`tensor` or :obj:`None`): The minimum value to be clipped to
            max_value (:obj:`scalar` or :obj:`tensor` or :obj:`None`): The maximum value to be clipped to

        Returns:
            NumPy ndarray: A clipped `tensor`
        """
        return np.clip(tensor_in, min_value, max_value)

    def erf(self, tensor_in: Tensor[T]) -> ArrayLike:
        """
        The error function of complex argument.

        Example:

            >>> import pyhf
            >>> pyhf.set_backend("numpy")
            >>> a = pyhf.tensorlib.astensor([-2., -1., 0., 1., 2.])
            >>> pyhf.tensorlib.erf(a)
            array([-0.99532227, -0.84270079,  0.        ,  0.84270079,  0.99532227])

        Args:
            tensor_in (:obj:`tensor`): The input tensor object

        Returns:
            NumPy ndarray: The values of the error function at the given points.
        """
        return special.erf(tensor_in)  # type: ignore[no-any-return]

    def erfinv(self, tensor_in: Tensor[T]) -> ArrayLike:
        """
        The inverse of the error function of complex argument.

        Example:

            >>> import pyhf
            >>> pyhf.set_backend("numpy")
            >>> a = pyhf.tensorlib.astensor([-2., -1., 0., 1., 2.])
            >>> pyhf.tensorlib.erfinv(pyhf.tensorlib.erf(a))
            array([-2., -1.,  0.,  1.,  2.])

        Args:
            tensor_in (:obj:`tensor`): The input tensor object

        Returns:
            NumPy ndarray: The values of the inverse of the error function at the given points.
        """
        return special.erfinv(tensor_in)  # type: ignore[no-any-return]

    def tile(self, tensor_in: Tensor[T], repeats: int | Sequence[int]) -> ArrayLike:
        """
        Repeat tensor data along a specific dimension

        Example:

            >>> import pyhf
            >>> pyhf.set_backend("numpy")
            >>> a = pyhf.tensorlib.astensor([[1.0], [2.0]])
            >>> pyhf.tensorlib.tile(a, (1, 2))
            array([[1., 1.],
                   [2., 2.]])

        Args:
            tensor_in (:obj:`tensor`): The tensor to be repeated
            repeats (:obj:`tensor`): The tuple of multipliers for each dimension

        Returns:
            NumPy ndarray: The tensor with repeated axes
        """
        return np.tile(tensor_in, repeats)

    def conditional(
        self,
        predicate: NDArray[np.bool_],
        true_callable: Callable[[], Tensor[T]],
        false_callable: Callable[[], Tensor[T]],
    ) -> ArrayLike:
        """
        Runs a callable conditional on the boolean value of the evaluation of a predicate

        Example:

            >>> import pyhf
            >>> pyhf.set_backend("numpy")
            >>> tensorlib = pyhf.tensorlib
            >>> a = tensorlib.astensor([4])
            >>> b = tensorlib.astensor([5])
            >>> tensorlib.conditional((a < b)[0], lambda: a + b, lambda: a - b)
            array([9.])

        Args:
            predicate (:obj:`scalar`): The logical condition that determines which callable to evaluate
            true_callable (:obj:`callable`): The callable that is evaluated when the :code:`predicate` evaluates to :code:`true`
            false_callable (:obj:`callable`): The callable that is evaluated when the :code:`predicate` evaluates to :code:`false`

        Returns:
            NumPy ndarray: The output of the callable that was evaluated
        """
        return true_callable() if predicate else false_callable()

    def tolist(self, tensor_in: Tensor[T] | list[T]) -> list[T]:
        try:
            return tensor_in.tolist()  # type: ignore[union-attr,no-any-return]
        except AttributeError:
            if isinstance(tensor_in, list):
                return tensor_in
            raise

    def outer(self, tensor_in_1: Tensor[T], tensor_in_2: Tensor[T]) -> ArrayLike:
        return np.outer(tensor_in_1, tensor_in_2)  # type: ignore[arg-type]

    def gather(self, tensor: Tensor[T], indices: NDArray[np.integer[T]]) -> ArrayLike:
        return tensor[indices]

    def boolean_mask(self, tensor: Tensor[T], mask: NDArray[np.bool_]) -> ArrayLike:
        return tensor[mask]

    def isfinite(self, tensor: Tensor[T]) -> NDArray[np.bool_]:
        return np.isfinite(tensor)

    def astensor(
        self, tensor_in: ArrayLike, dtype: FloatIntOrBool = 'float'
    ) -> ArrayLike:
        """
        Convert to a NumPy array.

        Example:

            >>> import pyhf
            >>> pyhf.set_backend("numpy")
            >>> tensor = pyhf.tensorlib.astensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            >>> tensor
            array([[1., 2., 3.],
                   [4., 5., 6.]])
            >>> type(tensor)
            <class 'numpy.ndarray'>

        Args:
            tensor_in (Number or Tensor): Tensor object

        Returns:
            `numpy.ndarray`: A multi-dimensional, fixed-size homogeneous array.
        """
        try:
            dtype_obj = self.dtypemap[dtype]
        except KeyError:
            log.error(
                'Invalid dtype: dtype must be float, int, or bool.', exc_info=True
            )
            raise

        return np.asarray(tensor_in, dtype=dtype_obj)

    def sum(self, tensor_in: Tensor[T], axis: int | None = None) -> ArrayLike:
        return np.sum(tensor_in, axis=axis)

    def product(self, tensor_in: Tensor[T], axis: Shape | None = None) -> ArrayLike:
        return np.prod(tensor_in, axis=axis)  # type: ignore[arg-type]

    def abs(self, tensor: Tensor[T]) -> ArrayLike:
        return np.abs(tensor)

    def ones(self, shape: Shape, dtype: FloatIntOrBool = "float") -> ArrayLike:
        try:
            dtype_obj = self.dtypemap[dtype]
        except KeyError:
            log.error(
                f"Invalid dtype: dtype must be one of {list(self.dtypemap.keys())}.",
                exc_info=True,
            )
            raise

        return np.ones(shape, dtype=dtype_obj)

    def zeros(self, shape: Shape, dtype: FloatIntOrBool = "float") -> ArrayLike:
        try:
            dtype_obj = self.dtypemap[dtype]
        except KeyError:
            log.error(
                f"Invalid dtype: dtype must be one of {list(self.dtypemap.keys())}.",
                exc_info=True,
            )
            raise

        return np.zeros(shape, dtype=dtype_obj)

    def power(self, tensor_in_1: Tensor[T], tensor_in_2: Tensor[T]) -> ArrayLike:
        return np.power(tensor_in_1, tensor_in_2)

    def sqrt(self, tensor_in: Tensor[T]) -> ArrayLike:
        return np.sqrt(tensor_in)

    def divide(self, tensor_in_1: Tensor[T], tensor_in_2: Tensor[T]) -> ArrayLike:
        return np.divide(tensor_in_1, tensor_in_2)

    def log(self, tensor_in: Tensor[T]) -> ArrayLike:
        return np.log(tensor_in)

    def exp(self, tensor_in: Tensor[T]) -> ArrayLike:
        return np.exp(tensor_in)

    def percentile(
        self,
        tensor_in: Tensor[T],
        q: Tensor[T],
        axis: None | Shape = None,
        interpolation: Literal[
            "linear", "lower", "higher", "midpoint", "nearest"
        ] = "linear",
    ) -> ArrayLike:
        r"""
        Compute the :math:`q`-th percentile of the tensor along the specified axis.

        Example:

            >>> import pyhf
            >>> pyhf.set_backend("numpy")
            >>> a = pyhf.tensorlib.astensor([[10, 7, 4], [3, 2, 1]])
            >>> pyhf.tensorlib.percentile(a, 50)
            3.5
            >>> pyhf.tensorlib.percentile(a, 50, axis=1)
            array([7., 2.])

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
            NumPy ndarray: The value of the :math:`q`-th percentile of the tensor along the specified axis.

        .. versionadded:: 0.7.0
        """
        # see https://github.com/numpy/numpy/issues/22125
        return np.percentile(tensor_in, q, axis=axis, interpolation=interpolation)  # type: ignore[call-overload,no-any-return]

    def stack(self, sequence: Sequence[Tensor[T]], axis: int = 0) -> ArrayLike:
        return np.stack(sequence, axis=axis)

    def where(
        self, mask: NDArray[np.bool_], tensor_in_1: Tensor[T], tensor_in_2: Tensor[T]
    ) -> ArrayLike:
        return np.where(mask, tensor_in_1, tensor_in_2)

    def concatenate(self, sequence: Tensor[T], axis: None | int = 0) -> ArrayLike:
        """
        Join a sequence of arrays along an existing axis.

        Args:
            sequence: sequence of tensors
            axis: dimension along which to concatenate

        Returns:
            output: the concatenated tensor

        """
        return np.concatenate(sequence, axis=axis)

    def simple_broadcast(self, *args: Sequence[Tensor[T]]) -> Sequence[Tensor[T]]:
        """
        Broadcast a sequence of 1 dimensional arrays.

        Example:

            >>> import pyhf
            >>> pyhf.set_backend("numpy")
            >>> pyhf.tensorlib.simple_broadcast(
            ...   pyhf.tensorlib.astensor([1]),
            ...   pyhf.tensorlib.astensor([2, 3, 4]),
            ...   pyhf.tensorlib.astensor([5, 6, 7]))
            [array([1., 1., 1.]), array([2., 3., 4.]), array([5., 6., 7.])]

        Args:
            args (Array of Tensors): Sequence of arrays

        Returns:
            list of Tensors: The sequence broadcast together.
        """
        return np.broadcast_arrays(*args)

    def shape(self, tensor: Tensor[T]) -> Shape:
        return tensor.shape

    def reshape(self, tensor: Tensor[T], newshape: Shape) -> ArrayLike:
        return np.reshape(tensor, newshape)

    def ravel(self, tensor: Tensor[T]) -> ArrayLike:
        """
        Return a flattened view of the tensor, not a copy.

        Example:

            >>> import pyhf
            >>> pyhf.set_backend("numpy")
            >>> tensor = pyhf.tensorlib.astensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            >>> pyhf.tensorlib.ravel(tensor)
            array([1., 2., 3., 4., 5., 6.])

        Args:
            tensor (Tensor): Tensor object

        Returns:
            `numpy.ndarray`: A flattened array.
        """
        return np.ravel(tensor)

    def einsum(self, subscripts: str, *operands: Sequence[Tensor[T]]) -> ArrayLike:
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
        return np.einsum(subscripts, *operands)  # type: ignore[arg-type,no-any-return]

    def poisson_logpdf(self, n: Tensor[T], lam: Tensor[T]) -> ArrayLike:
        return xlogy(n, lam) - lam - gammaln(n + 1.0)  # type: ignore[no-any-return]

    def poisson(self, n: Tensor[T], lam: Tensor[T]) -> ArrayLike:
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
            >>> pyhf.set_backend("numpy")
            >>> pyhf.tensorlib.poisson(5., 6.)
            0.16062314...
            >>> values = pyhf.tensorlib.astensor([5., 9.])
            >>> rates = pyhf.tensorlib.astensor([6., 8.])
            >>> pyhf.tensorlib.poisson(values, rates)
            array([0.16062314, 0.12407692])

        Args:
            n (:obj:`tensor` or :obj:`float`): The value at which to evaluate the approximation to the Poisson distribution p.m.f.
                                  (the observed number of events)
            lam (:obj:`tensor` or :obj:`float`): The mean of the Poisson distribution p.m.f.
                                    (the expected number of events)

        Returns:
            NumPy float: Value of the continuous approximation to Poisson(n|lam)
        """
        _n = np.asarray(n)
        _lam = np.asarray(lam)
        return np.exp(xlogy(_n, _lam) - _lam - gammaln(_n + 1.0))  # type: ignore[no-any-return,operator]

    def normal_logpdf(self, x: Tensor[T], mu: Tensor[T], sigma: Tensor[T]) -> ArrayLike:
        # this is much faster than
        # norm.logpdf(x, loc=mu, scale=sigma)
        # https://codereview.stackexchange.com/questions/69718/fastest-computation-of-n-likelihoods-on-normal-distributions
        root2 = np.sqrt(2)
        root2pi = np.sqrt(2 * np.pi)
        prefactor = -np.log(sigma * root2pi)
        summand = -np.square(np.divide((x - mu), (root2 * sigma)))
        return prefactor + summand  # type: ignore[no-any-return]

    # def normal_logpdf(self, x, mu, sigma):
    #     return norm.logpdf(x, loc=mu, scale=sigma)

    def normal(self, x: Tensor[T], mu: Tensor[T], sigma: Tensor[T]) -> ArrayLike:
        r"""
        The probability density function of the Normal distribution evaluated
        at :code:`x` given parameters of mean of :code:`mu` and standard deviation
        of :code:`sigma`.

        Example:

            >>> import pyhf
            >>> pyhf.set_backend("numpy")
            >>> pyhf.tensorlib.normal(0.5, 0., 1.)
            0.35206532...
            >>> values = pyhf.tensorlib.astensor([0.5, 2.0])
            >>> means = pyhf.tensorlib.astensor([0., 2.3])
            >>> sigmas = pyhf.tensorlib.astensor([1., 0.8])
            >>> pyhf.tensorlib.normal(values, means, sigmas)
            array([0.35206533, 0.46481887])

        Args:
            x (:obj:`tensor` or :obj:`float`): The value at which to evaluate the Normal distribution p.d.f.
            mu (:obj:`tensor` or :obj:`float`): The mean of the Normal distribution
            sigma (:obj:`tensor` or :obj:`float`): The standard deviation of the Normal distribution

        Returns:
            NumPy float: Value of Normal(x|mu, sigma)
        """
        return norm.pdf(x, loc=mu, scale=sigma)  # type: ignore[no-any-return]

    def normal_cdf(
        self, x: Tensor[T], mu: float | Tensor[T] = 0, sigma: float | Tensor[T] = 1
    ) -> ArrayLike:
        """
        The cumulative distribution function for the Normal distribution

        Example:

            >>> import pyhf
            >>> pyhf.set_backend("numpy")
            >>> pyhf.tensorlib.normal_cdf(0.8)
            0.78814460...
            >>> values = pyhf.tensorlib.astensor([0.8, 2.0])
            >>> pyhf.tensorlib.normal_cdf(values)
            array([0.7881446 , 0.97724987])

        Args:
            x (:obj:`tensor` or :obj:`float`): The observed value of the random variable to evaluate the CDF for
            mu (:obj:`tensor` or :obj:`float`): The mean of the Normal distribution
            sigma (:obj:`tensor` or :obj:`float`): The standard deviation of the Normal distribution

        Returns:
            NumPy float: The CDF
        """
        return norm.cdf(x, loc=mu, scale=sigma)  # type: ignore[no-any-return]

    def poisson_dist(self, rate: Tensor[T]) -> _BasicPoisson:
        r"""
        The Poisson distribution with rate parameter :code:`rate`.

        Example:
            >>> import pyhf
            >>> pyhf.set_backend("numpy")
            >>> rates = pyhf.tensorlib.astensor([5, 8])
            >>> values = pyhf.tensorlib.astensor([4, 9])
            >>> poissons = pyhf.tensorlib.poisson_dist(rates)
            >>> poissons.log_prob(values)
            array([-1.74030218, -2.0868536 ])

        Args:
            rate (:obj:`tensor` or :obj:`float`): The mean of the Poisson distribution (the expected number of events)

        Returns:
            Poisson distribution: The Poisson distribution class
        """
        return _BasicPoisson(rate)

    def normal_dist(self, mu: Tensor[T], sigma: Tensor[T]) -> _BasicNormal:
        r"""
        The Normal distribution with mean :code:`mu` and standard deviation :code:`sigma`.

        Example:
            >>> import pyhf
            >>> pyhf.set_backend("numpy")
            >>> means = pyhf.tensorlib.astensor([5, 8])
            >>> stds = pyhf.tensorlib.astensor([1, 0.5])
            >>> values = pyhf.tensorlib.astensor([4, 9])
            >>> normals = pyhf.tensorlib.normal_dist(means, stds)
            >>> normals.log_prob(values)
            array([-1.41893853, -2.22579135])

        Args:
            mu (:obj:`tensor` or :obj:`float`): The mean of the Normal distribution
            sigma (:obj:`tensor` or :obj:`float`): The standard deviation of the Normal distribution

        Returns:
            Normal distribution: The Normal distribution class

        """
        return _BasicNormal(mu, sigma)

    def to_numpy(self, tensor_in: Tensor[T]) -> ArrayLike:
        """
        Return the input tensor as it already is a :class:`numpy.ndarray`.
        This API exists only for ``pyhf.tensorlib`` compatibility.

        Example:
            >>> import pyhf
            >>> pyhf.set_backend("numpy")
            >>> tensor = pyhf.tensorlib.astensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            >>> tensor
            array([[1., 2., 3.],
                   [4., 5., 6.]])
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
        return tensor_in

    def transpose(self, tensor_in: Tensor[T]) -> ArrayLike:
        """
        Transpose the tensor.

        Example:
            >>> import pyhf
            >>> pyhf.set_backend("numpy")
            >>> tensor = pyhf.tensorlib.astensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            >>> tensor
            array([[1., 2., 3.],
                   [4., 5., 6.]])
            >>> pyhf.tensorlib.transpose(tensor)
            array([[1., 4.],
                   [2., 5.],
                   [3., 6.]])

        Args:
            tensor_in (:obj:`tensor`): The input tensor object.

        Returns:
            :class:`numpy.ndarray`: The transpose of the input tensor.

        .. versionadded:: 0.7.0
        """
        return tensor_in.transpose()
