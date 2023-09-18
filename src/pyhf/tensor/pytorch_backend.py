"""PyTorch Tensor Library Module."""
import torch
import torch.autograd
from torch.distributions.utils import broadcast_all
import logging
import math

log = logging.getLogger(__name__)


class pytorch_backend:
    """PyTorch backend for pyhf"""

    __slots__ = ['name', 'precision', 'dtypemap', 'default_do_grad']

    #: The array type for pytorch
    array_type = torch.Tensor

    #: The array content type for pytorch
    array_subtype = torch.Tensor

    def __init__(self, **kwargs):
        self.name = 'pytorch'
        self.precision = kwargs.get('precision', '64b')
        self.dtypemap = {
            'float': torch.float64 if self.precision == '64b' else torch.float32,
            'int': torch.int64 if self.precision == '64b' else torch.int32,
            'bool': torch.bool,
        }
        self.default_do_grad = True

    def _setup(self):
        """
        Run any global setups for the pytorch lib.
        """
        torch.set_default_dtype(self.dtypemap["float"])

    def clip(self, tensor_in, min_value, max_value):
        """
        Clips (limits) the tensor values to be within a specified min and max.

        Example:

            >>> import pyhf
            >>> pyhf.set_backend("pytorch")
            >>> a = pyhf.tensorlib.astensor([-2, -1, 0, 1, 2])
            >>> pyhf.tensorlib.clip(a, -1, 1)
            tensor([-1., -1.,  0.,  1.,  1.])

        Args:
            tensor_in (:obj:`tensor`): The input tensor object
            min_value (:obj:`scalar` or :obj:`tensor` or :obj:`None`): The minimum value to be clipped to
            max_value (:obj:`scalar` or :obj:`tensor` or :obj:`None`): The maximum value to be clipped to

        Returns:
            PyTorch tensor: A clipped `tensor`
        """
        return torch.clamp(tensor_in, min_value, max_value)

    def erf(self, tensor_in):
        """
        The error function of complex argument.

        Example:

            >>> import pyhf
            >>> pyhf.set_backend("pytorch")
            >>> a = pyhf.tensorlib.astensor([-2., -1., 0., 1., 2.])
            >>> pyhf.tensorlib.erf(a)
            tensor([-0.9953, -0.8427,  0.0000,  0.8427,  0.9953])

        Args:
            tensor_in (:obj:`tensor`): The input tensor object

        Returns:
            PyTorch Tensor: The values of the error function at the given points.
        """
        return torch.erf(tensor_in)

    def erfinv(self, tensor_in):
        """
        The inverse of the error function of complex argument.

        Example:

            >>> import pyhf
            >>> pyhf.set_backend("pytorch")
            >>> a = pyhf.tensorlib.astensor([-2., -1., 0., 1., 2.])
            >>> pyhf.tensorlib.erfinv(pyhf.tensorlib.erf(a))
            tensor([-2.0000, -1.0000,  0.0000,  1.0000,  2.0000])

        Args:
            tensor_in (:obj:`tensor`): The input tensor object

        Returns:
            PyTorch Tensor: The values of the inverse of the error function at the given points.
        """
        return torch.erfinv(tensor_in)

    def conditional(self, predicate, true_callable, false_callable):
        """
        Runs a callable conditional on the boolean value of the evaluation of a predicate

        Example:

            >>> import pyhf
            >>> pyhf.set_backend("pytorch")
            >>> tensorlib = pyhf.tensorlib
            >>> a = tensorlib.astensor([4])
            >>> b = tensorlib.astensor([5])
            >>> tensorlib.conditional((a < b)[0], lambda: a + b, lambda: a - b)
            tensor([9.])

        Args:
            predicate (:obj:`scalar`): The logical condition that determines which callable to evaluate
            true_callable (:obj:`callable`): The callable that is evaluated when the :code:`predicate` evaluates to :code:`true`
            false_callable (:obj:`callable`): The callable that is evaluated when the :code:`predicate` evaluates to :code:`false`

        Returns:
            PyTorch Tensor: The output of the callable that was evaluated
        """
        return true_callable() if predicate else false_callable()

    def tolist(self, tensor_in):
        try:
            return tensor_in.data.numpy().tolist()
        except AttributeError:
            if isinstance(tensor_in, list):
                return tensor_in
            raise

    def tile(self, tensor_in, repeats):
        """
        Repeat tensor data along a specific dimension

        Example:

            >>> import pyhf
            >>> pyhf.set_backend("pytorch")
            >>> a = pyhf.tensorlib.astensor([[1.0], [2.0]])
            >>> pyhf.tensorlib.tile(a, (1, 2))
            tensor([[1., 1.],
                    [2., 2.]])

        Args:
            tensor_in (:obj:`tensor`): The tensor to be repeated
            repeats (:obj:`tensor`): The tuple of multipliers for each dimension

        Returns:
            PyTorch tensor: The tensor with repeated axes
        """
        return tensor_in.tile(repeats)

    def outer(self, tensor_in_1, tensor_in_2):
        """
        Outer product of the input tensors.

        Example:

            >>> import pyhf
            >>> pyhf.set_backend("pytorch")
            >>> a = pyhf.tensorlib.astensor([1.0, 2.0, 3.0])
            >>> b = pyhf.tensorlib.astensor([1.0, 2.0, 3.0, 4.0])
            >>> pyhf.tensorlib.outer(a, b)
            tensor([[ 1.,  2.,  3.,  4.],
                    [ 2.,  4.,  6.,  8.],
                    [ 3.,  6.,  9., 12.]])

        Args:
            tensor_in_1 (:obj:`tensor`): 1-D input tensor.
            tensor_in_2 (:obj:`tensor`): 1-D input tensor.

        Returns:
            PyTorch tensor: The outer product.
        """
        return torch.outer(tensor_in_1, tensor_in_2)

    def astensor(self, tensor_in, dtype='float'):
        """
        Convert to a PyTorch Tensor.

        Example:

            >>> import pyhf
            >>> pyhf.set_backend("pytorch")
            >>> tensor = pyhf.tensorlib.astensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            >>> tensor
            tensor([[1., 2., 3.],
                    [4., 5., 6.]])
            >>> type(tensor)
            <class 'torch.Tensor'>

        Args:
            tensor_in (Number or Tensor): Tensor object

        Returns:
            torch.Tensor: A multi-dimensional matrix containing elements of a single data type.
        """
        try:
            dtype = self.dtypemap[dtype]
        except KeyError:
            log.error(
                'Invalid dtype: dtype must be float, int, or bool.', exc_info=True
            )
            raise

        return torch.as_tensor(tensor_in, dtype=dtype)

    def gather(self, tensor, indices):
        return tensor[indices.type(torch.LongTensor)]

    def boolean_mask(self, tensor, mask):
        return torch.masked_select(tensor, mask)

    def reshape(self, tensor, newshape):
        return torch.reshape(tensor, newshape)

    def shape(self, tensor):
        return tuple(map(int, tensor.shape))

    def ravel(self, tensor):
        """
        Return a flattened view of the tensor, not a copy.

        Example:

            >>> import pyhf
            >>> pyhf.set_backend("pytorch")
            >>> tensor = pyhf.tensorlib.astensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            >>> pyhf.tensorlib.ravel(tensor)
            tensor([1., 2., 3., 4., 5., 6.])

        Args:
            tensor (Tensor): Tensor object

        Returns:
            `torch.Tensor`: A flattened array.
        """
        return torch.ravel(tensor)

    def sum(self, tensor_in, axis=None):
        return (
            torch.sum(tensor_in)
            if (axis is None or tensor_in.shape == torch.Size([]))
            else torch.sum(tensor_in, axis)
        )

    def product(self, tensor_in, axis=None):
        return torch.prod(tensor_in) if axis is None else torch.prod(tensor_in, axis)

    def abs(self, tensor):
        return torch.abs(tensor)

    def ones(self, shape, dtype="float"):
        try:
            dtype = self.dtypemap[dtype]
        except KeyError:
            log.error(
                f"Invalid dtype: dtype must be one of {list(self.dtypemap.keys())}.",
                exc_info=True,
            )
            raise

        return torch.ones(shape, dtype=dtype)

    def zeros(self, shape, dtype="float"):
        try:
            dtype = self.dtypemap[dtype]
        except KeyError:
            log.error(
                f"Invalid dtype: dtype must be one of {list(self.dtypemap.keys())}.",
                exc_info=True,
            )
            raise

        return torch.zeros(shape, dtype=dtype)

    def power(self, tensor_in_1, tensor_in_2):
        return torch.pow(tensor_in_1, tensor_in_2)

    def sqrt(self, tensor_in):
        return torch.sqrt(tensor_in)

    def divide(self, tensor_in_1, tensor_in_2):
        return torch.div(tensor_in_1, tensor_in_2)

    def log(self, tensor_in):
        return torch.log(tensor_in)

    def exp(self, tensor_in):
        return torch.exp(tensor_in)

    def percentile(self, tensor_in, q, axis=None, interpolation="linear"):
        r"""
        Compute the :math:`q`-th percentile of the tensor along the specified axis.

        Example:

            >>> import pyhf
            >>> pyhf.set_backend("pytorch")
            >>> a = pyhf.tensorlib.astensor([[10, 7, 4], [3, 2, 1]])
            >>> pyhf.tensorlib.percentile(a, 50)
            tensor(3.5000)
            >>> pyhf.tensorlib.percentile(a, 50, axis=1)
            tensor([7., 2.])

        Args:
            tensor_in (`tensor`): The tensor containing the data
            q (:obj:`float` or `tensor`): The :math:`q`-th percentile to compute
            axis (`number` or `tensor`): The dimensions along which to compute
            interpolation (:obj:`str`): The interpolation method to use when the
             desired percentile lies between two data points ``i < j``:

                - ``'linear'``: ``i + (j - i) * fraction``, where ``fraction`` is the
                  fractional part of the index surrounded by ``i`` and ``j``.

                - ``'lower'``: Not yet implemented in PyTorch.

                - ``'higher'``: Not yet implemented in PyTorch.

                - ``'midpoint'``: Not yet implemented in PyTorch.

                - ``'nearest'``: Not yet implemented in PyTorch.

        Returns:
            PyTorch tensor: The value of the :math:`q`-th percentile of the tensor along the specified axis.

        .. versionadded:: 0.7.0
        """
        # Interpolation options not yet supported
        # c.f. https://github.com/pytorch/pytorch/pull/49267
        # c.f. https://github.com/pytorch/pytorch/pull/59397
        return torch.quantile(tensor_in, q / 100, dim=axis)

    def stack(self, sequence, axis=0):
        return torch.stack(sequence, dim=axis)

    def where(self, mask, tensor_in_1, tensor_in_2):
        return torch.where(mask, tensor_in_1, tensor_in_2)

    def concatenate(self, sequence, axis=0):
        """
        Join a sequence of arrays along an existing axis.

        Args:
            sequence: sequence of tensors
            axis: dimension along which to concatenate

        Returns:
            output: the concatenated tensor

        """
        return torch.cat(sequence, dim=axis)

    def isfinite(self, tensor):
        return torch.isfinite(tensor)

    def simple_broadcast(self, *args):
        """
        Broadcast a sequence of 1 dimensional arrays.

        Example:

            >>> import pyhf
            >>> pyhf.set_backend("pytorch")
            >>> pyhf.tensorlib.simple_broadcast(
            ...   pyhf.tensorlib.astensor([1]),
            ...   pyhf.tensorlib.astensor([2, 3, 4]),
            ...   pyhf.tensorlib.astensor([5, 6, 7]))
            [tensor([1., 1., 1.]), tensor([2., 3., 4.]), tensor([5., 6., 7.])]

        Args:
            args (Array of Tensors): Sequence of arrays

        Returns:
            list of Tensors: The sequence broadcast together.
        """

        args = [arg.view(1) if not self.shape(arg) else arg for arg in args]
        max_dim = max(map(len, args))
        try:
            assert not [arg for arg in args if 1 < len(arg) < max_dim]
        except AssertionError:
            log.error(
                'ERROR: The arguments must be of compatible size: 1 or %i',
                max_dim,
                exc_info=True,
            )
            raise

        broadcast = [arg if len(arg) > 1 else arg.expand(max_dim) for arg in args]
        return broadcast

    def einsum(self, subscripts, *operands):
        """
        This function provides a way of computing multilinear expressions (i.e.
        sums of products) using the Einstein summation convention.

        Args:
            subscripts: str, specifies the subscripts for summation
            operands: list of array_like, these are the tensors for the operation

        Returns:
            tensor: the calculation based on the Einstein summation convention
        """
        return torch.einsum(subscripts, operands)

    def poisson_logpdf(self, n, lam):
        # validate_args=True disallows continuous approximation
        return torch.distributions.Poisson(lam, validate_args=False).log_prob(n)

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
            >>> pyhf.set_backend("pytorch")
            >>> pyhf.tensorlib.poisson(5., 6.)
            tensor(0.1606)
            >>> values = pyhf.tensorlib.astensor([5., 9.])
            >>> rates = pyhf.tensorlib.astensor([6., 8.])
            >>> pyhf.tensorlib.poisson(values, rates)
            tensor([0.1606, 0.1241])

        Args:
            n (:obj:`tensor` or :obj:`float`): The value at which to evaluate the approximation to the Poisson distribution p.m.f.
                                  (the observed number of events)
            lam (:obj:`tensor` or :obj:`float`): The mean of the Poisson distribution p.m.f.
                                    (the expected number of events)

        Returns:
            PyTorch FloatTensor: Value of the continuous approximation to Poisson(n|lam)
        """
        # validate_args=True disallows continuous approximation
        return torch.exp(
            torch.distributions.Poisson(lam, validate_args=False).log_prob(n)
        )

    def normal_logpdf(self, x, mu, sigma):
        x = self.astensor(x)
        mu = self.astensor(mu)
        sigma = self.astensor(sigma)

        normal = torch.distributions.Normal(mu, sigma)
        return normal.log_prob(x)

    def normal(self, x, mu, sigma):
        r"""
        The probability density function of the Normal distribution evaluated
        at :code:`x` given parameters of mean of :code:`mu` and standard deviation
        of :code:`sigma`.

        Example:

            >>> import pyhf
            >>> pyhf.set_backend("pytorch")
            >>> pyhf.tensorlib.normal(0.5, 0., 1.)
            tensor(0.3521)
            >>> values = pyhf.tensorlib.astensor([0.5, 2.0])
            >>> means = pyhf.tensorlib.astensor([0., 2.3])
            >>> sigmas = pyhf.tensorlib.astensor([1., 0.8])
            >>> pyhf.tensorlib.normal(values, means, sigmas)
            tensor([0.3521, 0.4648])

        Args:
            x (:obj:`tensor` or :obj:`float`): The value at which to evaluate the Normal distribution p.d.f.
            mu (:obj:`tensor` or :obj:`float`): The mean of the Normal distribution
            sigma (:obj:`tensor` or :obj:`float`): The standard deviation of the Normal distribution

        Returns:
            PyTorch FloatTensor: Value of Normal(x|mu, sigma)
        """
        x = self.astensor(x)
        mu = self.astensor(mu)
        sigma = self.astensor(sigma)

        normal = torch.distributions.Normal(mu, sigma)
        return self.exp(normal.log_prob(x))

    def normal_cdf(self, x, mu=0.0, sigma=1.0):
        """
        The cumulative distribution function for the Normal distribution

        Example:

            >>> import pyhf
            >>> pyhf.set_backend("pytorch")
            >>> pyhf.tensorlib.normal_cdf(0.8)
            tensor(0.7881)
            >>> values = pyhf.tensorlib.astensor([0.8, 2.0])
            >>> pyhf.tensorlib.normal_cdf(values)
            tensor([0.7881, 0.9772])

        Args:
            x (:obj:`tensor` or :obj:`float`): The observed value of the random variable to evaluate the CDF for
            mu (:obj:`tensor` or :obj:`float`): The mean of the Normal distribution
            sigma (:obj:`tensor` or :obj:`float`): The standard deviation of the Normal distribution

        Returns:
            PyTorch FloatTensor: The CDF
        """
        # the implementation of torch.Normal.cdf uses torch.erf:
        # 0.5 * (1 + torch.erf((value - self.loc) * self.scale.reciprocal() / math.sqrt(2)))
        # (see https://github.com/pytorch/pytorch/blob/3bbedb34b9b316729a27e793d94488b574e1577a/torch/distributions/normal.py#L78-L81)
        # we get a more numerically stable variant for low p-values/high significances using erfc(x) := 1 - erf(x)
        # since erf(-x) = -erf(x) we can replace
        # 1 + erf(x) = 1 - erf(-x) = 1 - (1 - erfc(-x)) = erfc(-x)
        mu, sigma = broadcast_all(mu, sigma)
        return 0.5 * torch.erfc(-((x - mu) * sigma.reciprocal() / math.sqrt(2)))

    def poisson_dist(self, rate):
        r"""
        The Poisson distribution with rate parameter :code:`rate`.

        Example:
            >>> import pyhf
            >>> pyhf.set_backend("pytorch")
            >>> rates = pyhf.tensorlib.astensor([5, 8])
            >>> values = pyhf.tensorlib.astensor([4, 9])
            >>> poissons = pyhf.tensorlib.poisson_dist(rates)
            >>> poissons.log_prob(values)
            tensor([-1.7403, -2.0869])

        Args:
            rate (:obj:`tensor` or :obj:`float`): The mean of the Poisson distribution (the expected number of events)

        Returns:
            PyTorch Poisson distribution: The Poisson distribution class

        """
        # validate_args=True disallows continuous approximation
        return torch.distributions.Poisson(rate, validate_args=False)

    def normal_dist(self, mu, sigma):
        r"""
        The Normal distribution with mean :code:`mu` and standard deviation :code:`sigma`.

        Example:
            >>> import pyhf
            >>> pyhf.set_backend("pytorch")
            >>> means = pyhf.tensorlib.astensor([5, 8])
            >>> stds = pyhf.tensorlib.astensor([1, 0.5])
            >>> values = pyhf.tensorlib.astensor([4, 9])
            >>> normals = pyhf.tensorlib.normal_dist(means, stds)
            >>> normals.log_prob(values)
            tensor([-1.4189, -2.2258])

        Args:
            mu (:obj:`tensor` or :obj:`float`): The mean of the Normal distribution
            sigma (:obj:`tensor` or :obj:`float`): The standard deviation of the Normal distribution

        Returns:
            PyTorch Normal distribution: The Normal distribution class

        """
        return torch.distributions.Normal(mu, sigma)

    def to_numpy(self, tensor_in):
        """
        Convert the PyTorch tensor to a :class:`numpy.ndarray`.

        Example:
            >>> import pyhf
            >>> pyhf.set_backend("pytorch")
            >>> tensor = pyhf.tensorlib.astensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            >>> tensor
            tensor([[1., 2., 3.],
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
        return tensor_in.numpy()

    def transpose(self, tensor_in):
        """
        Transpose the tensor.

        Example:
            >>> import pyhf
            >>> pyhf.set_backend("pytorch")
            >>> tensor = pyhf.tensorlib.astensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            >>> tensor
            tensor([[1., 2., 3.],
                    [4., 5., 6.]])
            >>> pyhf.tensorlib.transpose(tensor)
            tensor([[1., 4.],
                    [2., 5.],
                    [3., 6.]])

        Args:
            tensor_in (:obj:`tensor`): The input tensor object.

        Returns:
            PyTorch FloatTensor: The transpose of the input tensor.

        .. versionadded:: 0.7.0
        """
        return tensor_in.transpose(0, 1)
