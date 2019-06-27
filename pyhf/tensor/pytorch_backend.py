import torch
import torch.autograd
import logging

log = logging.getLogger(__name__)


class pytorch_backend(object):
    """PyTorch backend for pyhf"""

    def __init__(self, **kwargs):
        self.name = 'pytorch'

    def clip(self, tensor_in, min_value, max_value):
        """
        Clips (limits) the tensor values to be within a specified min and max.

        Example:

            >>> import pyhf
            >>> pyhf.set_backend(pyhf.tensor.pytorch_backend())
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
        tensor_in = self.astensor(tensor_in)
        return torch.clamp(tensor_in, min_value, max_value)

    def tolist(self, tensor_in):
        try:
            return tensor_in.data.numpy().tolist()
        except AttributeError:
            if isinstance(tensor_in, list):
                return tensor_in
            raise

    def outer(self, tensor_in_1, tensor_in_2):
        tensor_in_1 = self.astensor(tensor_in_1)
        tensor_in_2 = self.astensor(tensor_in_2)
        return torch.ger(tensor_in_1, tensor_in_2)

    def astensor(self, tensor_in, dtype='float'):
        """
        Convert to a PyTorch Tensor.

        Args:
            tensor_in (Number or Tensor): Tensor object

        Returns:
            torch.Tensor: A multi-dimensional matrix containing elements of a single data type.
        """
        dtypemap = {'float': torch.float, 'int': torch.int, 'bool': torch.uint8}
        try:
            dtype = dtypemap[dtype]
        except KeyError:
            log.error('Invalid dtype: dtype must be float, int, or bool.')
            raise

        tensor = torch.as_tensor(tensor_in, dtype=dtype)
        # Ensure non-empty tensor shape for consistency
        try:
            tensor.shape[0]
        except IndexError:
            tensor = tensor.expand(1)
        return tensor

    def gather(self, tensor, indices):
        return torch.take(tensor, indices.type(torch.LongTensor))

    def boolean_mask(self, tensor, mask):
        mask = self.astensor(mask).type(torch.ByteTensor)
        return torch.masked_select(tensor, mask)

    def reshape(self, tensor, newshape):
        return torch.reshape(tensor, newshape)

    def shape(self, tensor):
        return tuple(map(int, tensor.shape))

    def sum(self, tensor_in, axis=None):
        tensor_in = self.astensor(tensor_in)
        return (
            torch.sum(tensor_in)
            if (axis is None or tensor_in.shape == torch.Size([]))
            else torch.sum(tensor_in, axis)
        )

    def product(self, tensor_in, axis=None):
        tensor_in = self.astensor(tensor_in)
        return torch.prod(tensor_in) if axis is None else torch.prod(tensor_in, axis)

    def abs(self, tensor):
        tensor = self.astensor(tensor)
        return torch.abs(tensor)

    def ones(self, shape):
        return torch.Tensor(torch.ones(shape))

    def zeros(self, shape):
        return torch.Tensor(torch.zeros(shape))

    def power(self, tensor_in_1, tensor_in_2):
        tensor_in_1 = self.astensor(tensor_in_1)
        tensor_in_2 = self.astensor(tensor_in_2)
        return torch.pow(tensor_in_1, tensor_in_2)

    def sqrt(self, tensor_in):
        tensor_in = self.astensor(tensor_in)
        return torch.sqrt(tensor_in)

    def divide(self, tensor_in_1, tensor_in_2):
        tensor_in_1 = self.astensor(tensor_in_1)
        tensor_in_2 = self.astensor(tensor_in_2)
        return torch.div(tensor_in_1, tensor_in_2)

    def log(self, tensor_in):
        tensor_in = self.astensor(tensor_in)
        return torch.log(tensor_in)

    def exp(self, tensor_in):
        tensor_in = self.astensor(tensor_in)
        return torch.exp(tensor_in)

    def stack(self, sequence, axis=0):
        return torch.stack(sequence, dim=axis)

    def where(self, mask, tensor_in_1, tensor_in_2):
        mask = self.astensor(mask).type(torch.FloatTensor)
        tensor_in_1 = self.astensor(tensor_in_1)
        tensor_in_2 = self.astensor(tensor_in_2)
        return mask * tensor_in_1 + (1 - mask) * tensor_in_2

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
            >>> pyhf.set_backend(pyhf.tensor.pytorch_backend())
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

        args = [self.astensor(arg) for arg in args]
        max_dim = max(map(len, args))
        try:
            assert len([arg for arg in args if 1 < len(arg) < max_dim]) == 0
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
        ops = tuple(self.astensor(op) for op in operands)
        return torch.einsum(subscripts, ops)

    def poisson_logpdf(self, n, lam):
        n = self.astensor(n)
        lam = self.astensor(lam)
        return torch.distributions.Poisson(lam).log_prob(n)

    def poisson(self, n, lam):
        r"""
        The continous approximation, using :math:`n! = \Gamma\left(n+1\right)`,
        to the probability mass function of the Poisson distribution evaluated
        at :code:`n` given the parameter :code:`lam`.

        Example:

            >>> import pyhf
            >>> pyhf.set_backend(pyhf.tensor.pytorch_backend())
            >>> pyhf.tensorlib.poisson([5.], [6.])
            tensor([0.1606])
            >>> pyhf.tensorlib.poisson(5., 6.)
            tensor([0.1606])

        Args:
            n (`tensor` or `float`): The value at which to evaluate the approximation to the Poisson distribution p.m.f.
                                  (the observed number of events)
            lam (`tensor` or `float`): The mean of the Poisson distribution p.m.f.
                                    (the expected number of events)

        Returns:
            PyTorch FloatTensor: Value of the continous approximation to Poisson(n|lam)
        """
        n = self.astensor(n)
        lam = self.astensor(lam)
        return torch.exp(torch.distributions.Poisson(lam).log_prob(n))

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
            >>> pyhf.set_backend(pyhf.tensor.pytorch_backend())
            >>> pyhf.tensorlib.normal([0.5], [0.], [1.])
            tensor([0.3521])
            >>> pyhf.tensorlib.normal(0.5, 0., 1.)
            tensor([0.3521])

        Args:
            x (`tensor` or `float`): The value at which to evaluate the Normal distribution p.d.f.
            mu (`tensor` or `float`): The mean of the Normal distribution
            sigma (`tensor` or `float`): The standard deviation of the Normal distribution

        Returns:
            PyTorch FloatTensor: Value of Normal(x|mu, sigma)
        """
        x = self.astensor(x)
        mu = self.astensor(mu)
        sigma = self.astensor(sigma)
        normal = torch.distributions.Normal(mu, sigma)
        return self.exp(normal.log_prob(x))

    def normal_cdf(self, x, mu=[0.0], sigma=[1.0]):
        """
        The cumulative distribution function for the Normal distribution

        Example:

            >>> import pyhf
            >>> pyhf.set_backend(pyhf.tensor.pytorch_backend())
            >>> pyhf.tensorlib.normal_cdf([0.8])
            tensor([0.7881])

        Args:
            x (`tensor` or `float`): The observed value of the random variable to evaluate the CDF for
            mu (`tensor` or `float`): The mean of the Normal distribution
            sigma (`tensor` or `float`): The standard deviation of the Normal distribution

        Returns:
            PyTorch FloatTensor: The CDF
        """
        x = self.astensor(x)
        mu = self.astensor(mu)
        sigma = self.astensor(sigma)
        normal = torch.distributions.Normal(mu, sigma)
        return normal.cdf(x)
