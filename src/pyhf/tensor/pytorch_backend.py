"""PyTorch Tensor Library Module."""
import torch
import torch.autograd
from torch.distributions.utils import broadcast_all
import logging
import math

log = logging.getLogger(__name__)


class pytorch_backend(object):
    """PyTorch backend for pyhf"""

    __slots__ = ['name', 'precision', 'dtypemap', 'default_do_grad']

    def __init__(self, **kwargs):
        self.name = 'pytorch'
        self.precision = kwargs.get('precision', '32b')
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
            tensor_in (`tensor`): The input tensor object
            min_value (`scalar` or `tensor` or `None`): The minimum value to be cliped to
            max_value (`scalar` or `tensor` or `None`): The maximum value to be cliped to

        Returns:
            PyTorch tensor: A clipped `tensor`
        """
        return torch.clamp(tensor_in, min_value, max_value)

    def conditional(self, predicate, true_callable, false_callable):
        """
        Runs a callable conditional on the boolean value of the evaulation of a predicate

        Example:

            >>> import pyhf
            >>> pyhf.set_backend("pytorch")
            >>> tensorlib = pyhf.tensorlib
            >>> a = tensorlib.astensor([4])
            >>> b = tensorlib.astensor([5])
            >>> tensorlib.conditional((a < b)[0], lambda: a + b, lambda: a - b)
            tensor([9.])

        Args:
            predicate (`scalar`): The logical condition that determines which callable to evaluate
            true_callable (`callable`): The callable that is evaluated when the :code:`predicate` evalutes to :code:`true`
            false_callable (`callable`): The callable that is evaluated when the :code:`predicate` evalutes to :code:`false`

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
            tensor_in (`Tensor`): The tensor to be repeated
            repeats (`Tensor`): The tuple of multipliers for each dimension

        Returns:
            PyTorch tensor: The tensor with repeated axes
        """
        return tensor_in.repeat(repeats)

    def outer(self, tensor_in_1, tensor_in_2):
        return torch.ger(tensor_in_1, tensor_in_2)

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
            log.error('Invalid dtype: dtype must be float, int, or bool.')
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

    def ones(self, shape):
        return torch.ones(shape, dtype=self.dtypemap['float'])

    def zeros(self, shape):
        return torch.zeros(shape, dtype=self.dtypemap['float'])

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
        except AssertionError as error:
            log.error(
                'ERROR: The arguments must be of compatible size: 1 or %i', max_dim
            )
            raise error

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
        return torch.distributions.Poisson(lam).log_prob(n)

    def poisson(self, n, lam):
        r"""
        The continous approximation, using :math:`n! = \Gamma\left(n+1\right)`,
        to the probability mass function of the Poisson distribution evaluated
        at :code:`n` given the parameter :code:`lam`.

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
            n (`tensor` or `float`): The value at which to evaluate the approximation to the Poisson distribution p.m.f.
                                  (the observed number of events)
            lam (`tensor` or `float`): The mean of the Poisson distribution p.m.f.
                                    (the expected number of events)

        Returns:
            PyTorch FloatTensor: Value of the continous approximation to Poisson(n|lam)
        """
        return torch.exp(torch.distributions.Poisson(lam).log_prob(n))

    def normal_logpdf(self, x, mu, sigma):
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
            x (`tensor` or `float`): The value at which to evaluate the Normal distribution p.d.f.
            mu (`tensor` or `float`): The mean of the Normal distribution
            sigma (`tensor` or `float`): The standard deviation of the Normal distribution

        Returns:
            PyTorch FloatTensor: Value of Normal(x|mu, sigma)
        """
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
            x (`tensor` or `float`): The observed value of the random variable to evaluate the CDF for
            mu (`tensor` or `float`): The mean of the Normal distribution
            sigma (`tensor` or `float`): The standard deviation of the Normal distribution

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
            rate (`tensor` or `float`): The mean of the Poisson distribution (the expected number of events)

        Returns:
            PyTorch Poisson distribution: The Poisson distribution class

        """
        return torch.distributions.Poisson(rate)

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
            mu (`tensor` or `float`): The mean of the Normal distribution
            sigma (`tensor` or `float`): The standard deviation of the Normal distribution

        Returns:
            PyTorch Normal distribution: The Normal distribution class

        """
        return torch.distributions.Normal(mu, sigma)
