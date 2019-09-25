from mxnet import nd

import logging
import math  # Required for normal()
from numbers import Number  # Required for normal()
from scipy.stats import norm  # Required for normal_cdf()

log = logging.getLogger(__name__)


class mxnet_backend(object):
    """MXNet backend for pyhf"""

    def __init__(self, **kwargs):
        self.name = 'mxnet'

    def clip(self, tensor_in, min_value, max_value):
        """
        Clips (limits) the tensor values to be within a specified min and max.

        Example:

            >>> import pyhf
            >>> pyhf.set_backend(pyhf.tensor.mxnet_backend())
            >>> a = pyhf.tensorlib.astensor([-2, -1, 0, 1, 2])
            >>> pyhf.tensorlib.clip(a, -1, 1)
            <BLANKLINE>
            [-1. -1.  0.  1.  1.]
            <NDArray 5 @cpu(0)>

        Args:
            tensor_in (`tensor`): The input tensor object
            min_value (`scalar` or `tensor` or `None`): The minimum value to be cliped to
            max_value (`scalar` or `tensor` or `None`): The maximum value to be cliped to

        Returns:
            MXNet NDArray: A clipped `tensor`
        """
        return nd.clip(tensor_in, min_value, max_value)

    def tile(self, tensor_in, repeats):
        """
        Repeat tensor data along a specific dimension

        Example:

            >>> import pyhf
            >>> pyhf.set_backend(pyhf.tensor.mxnet_backend())
            >>> a = pyhf.tensorlib.astensor([[1.0], [2.0]])
            >>> pyhf.tensorlib.tile(a, (1, 2))
            <BLANKLINE>
            [[1. 1.]
             [2. 2.]]
            <NDArray 2x2 @cpu(0)>

        Args:
            tensor_in (`Tensor`): The tensor to be repeated
            repeats (`Tensor`): The tuple of multipliers for each dimension

        Returns:
            MXNet NDArray: The tensor with repeated axes
        """
        return nd.tile(tensor_in, repeats)

    def tolist(self, tensor_in):
        """
        Convert a tensor to a list.

        Args:
            tensor_in (Tensor): Input MXNet tensor

        Returns:
            list: The possibly nested list of tensor elements.
        """
        try:
            return tensor_in.asnumpy().tolist()
        except AttributeError:
            if isinstance(tensor_in, list):
                return tensor_in
            raise

    def outer(self, tensor_in_1, tensor_in_2):
        """
        The outer product of two tensors.

        Args:
            tensor_in_1 (Tensor): Tensor object
            tensor_in_2 (Tensor): Tensor object

        Returns:
            MXNet NDArray: The outer product.
        """
        tensor_1_shape = tensor_in_1.shape
        tensor_2_shape = tensor_in_2.shape
        if len(tensor_1_shape) == 1:
            tensor_1_shape = (tensor_1_shape[0], 1)
        if len(tensor_2_shape) == 1:
            tensor_2_shape = (tensor_2_shape[0], 1)

        rows1, cols1 = tensor_1_shape
        rows2, cols2 = tensor_2_shape
        return nd.reshape(
            nd.broadcast_mul(
                tensor_in_1.reshape((rows1, 1, cols1, 1)),
                tensor_in_2.reshape((1, rows2, 1, cols2)),
            ),
            (rows1 * cols1, rows2 * cols2),
        )

    def gather(self, tensor, indices):
        return tensor[indices]

    def boolean_mask(self, tensor, mask):
        raise NotImplementedError("mxnet::boolean_mask is not implemented.")

    def astensor(self, tensor_in, dtype='float'):
        """
        Convert to a MXNet NDArray.

        Args:
            tensor_in (Number or Tensor): Tensor object

        Returns:
            MXNet NDArray: A multi-dimensional, fixed-size homogenous array.
        """
        dtypemap = {'float': 'float32', 'int': 'int32', 'bool': 'uint8'}
        try:
            dtype = dtypemap[dtype]
        except KeyError:
            log.error('Invalid dtype: dtype must be float, int, or bool.')
            raise

        try:
            tensor = nd.array(tensor_in, dtype=dtype)
            # Ensure non-empty tensor shape for consistency
            try:
                tensor.shape[0]
            except IndexError:
                tensor = tensor.broadcast_to((1,))
        except ValueError:
            tensor = nd.array([tensor_in], dtype=dtype)
        return tensor

    def sum(self, tensor_in, axis=None):
        """
        Compute the sum of array elements over given axes.

        Args:
            tensor_in (Tensor): Tensor object
            axis (Number): The axes over which to sum

        Returns:
            MXNet NDArray: ndarray of the sum over the axes.
        """
        if axis is None or tensor_in.shape == nd.array([]).size:
            return nd.sum(tensor_in)
        return nd.sum(tensor_in, axis)

    def product(self, tensor_in, axis=None):
        """
        Product of array elements over given axes.

        Args:
            tensor_in (Tensor): Tensor object
            axis (Number): The axes over which to take the product

        Returns:
            MXNet NDArray: ndarray of the product over the axes.
        """
        if axis is None:
            return nd.prod(tensor_in)
        return nd.prod(tensor_in, axis)

    def abs(self, tensor):
        return nd.abs(tensor)

    def ones(self, shape):
        """
        A new array filled with all ones, with the given shape.

        Args:
            shape (Number): The shape of the array

        Returns:
            MXNet NDArray: ndarray of 1's with given shape.
        """
        return nd.ones(shape)

    def zeros(self, shape):
        """
        A new array filled with all zeros, with the given shape.

        Args:
            shape (Number): The shape of the array

        Returns:
            MXNet NDArray: ndarray of 0's with given shape.
        """
        return nd.zeros(shape)

    def power(self, tensor_in_1, tensor_in_2):
        """
        Result of first array elements raised to powers from second array,
        element-wise with broadcasting.

        Args:
            tensor_in_1 (Tensor): Tensor object
            tensor_in_2 (Tensor): Tensor object

        Returns:
            MXNet NDArray: First array elements raised to powers from second array.
        """
        return nd.power(tensor_in_1, tensor_in_2)

    def sqrt(self, tensor_in):
        """
        Element-wise square-root value of the input.

        Args:
            tensor_in (Tensor): Tensor object

        Returns:
            MXNet NDArray: Element-wise square-root value.
        """
        return nd.sqrt(tensor_in)

    def divide(self, tensor_in_1, tensor_in_2):
        """
        Element-wise division of the input arrays with broadcasting.

        Args:
            tensor_in_1 (Tensor): Tensor object
            tensor_in_2 (Tensor): Tensor object

        Returns:
            MXNet NDArray: Element-wise division of the input arrays.
        """
        return nd.divide(tensor_in_1, tensor_in_2)

    def log(self, tensor_in):
        """
        Element-wise Natural logarithmic value of the input.

        Args:
            tensor_in (Tensor): Tensor object

        Returns:
            MXNet NDArray: Element-wise Natural logarithmic value.
        """
        return nd.log(tensor_in)

    def exp(self, tensor_in):
        """
        Element-wise exponential value of the input.

        Args:
            tensor_in (Tensor): Tensor object

        Returns:
            MXNet NDArray: Element-wise exponential value.
        """
        return nd.exp(tensor_in)

    def stack(self, sequence, axis=0):
        """
        Join a sequence of arrays along a new axis.

        The axis parameter specifies the index of the new axis in the dimensions
        of the result. For example, if axis=0 it will be the first dimension and
        if axis=-1 it will be the last dimension.

        Args:
            sequence (Array of Tensors): Sequence of arrays
            axis (Number): The axis along which to join the arrays

        Returns:
            MXNet NDArray: ndarray comprised of the elements of the sequence.
        """
        return nd.stack(*sequence, axis=axis)

    def where(self, mask, tensor_in_1, tensor_in_2):
        """
        Apply a boolean selection mask to the elements of the input tensors.

        Example:

            >>> import pyhf
            >>> pyhf.set_backend(pyhf.tensor.mxnet_backend())
            >>> pyhf.tensorlib.where(
            ...   pyhf.tensorlib.astensor([1, 0, 1]),
            ...   pyhf.tensorlib.astensor([1, 1, 1]),
            ...   pyhf.tensorlib.astensor([2, 2, 2]))
            ...
            <BLANKLINE>
            [1. 2. 1.]
            <NDArray 3 @cpu(0)>

        Args:
            mask (bool): Boolean mask (boolean or tensor object of booleans)
            tensor_in_1 (Tensor): Tensor object
            tensor_in_2 (Tensor): Tensor object

        Returns:
            MXNet NDArray: The result of the mask being applied to the tensors.
        """
        return nd.where(mask, tensor_in_1, tensor_in_2)

    def concatenate(self, sequence, axis=0):
        """
        Join the elements of the sequence.

        Args:
            sequence (Array of Tensors): The sequence of arrays to join
            axis: dimension along which to concatenate

        Returns:
            MXNet NDArray: The ndarray of the joined elements.
        """
        return nd.concat(*sequence, dim=axis)

    def simple_broadcast(self, *args):
        """
        Broadcast a sequence of 1 dimensional arrays.

        Example:

            >>> import pyhf
            >>> pyhf.set_backend(pyhf.tensor.mxnet_backend())
            >>> pyhf.tensorlib.simple_broadcast(
            ...   pyhf.tensorlib.astensor([1]),
            ...   pyhf.tensorlib.astensor([2, 3, 4]),
            ...   pyhf.tensorlib.astensor([5, 6, 7]))
            <BLANKLINE>
            [[1. 1. 1.]
             [2. 3. 4.]
             [5. 6. 7.]]
            <NDArray 3x3 @cpu(0)>

        Args:
            args (Array of Tensors): Sequence of arrays

        Returns:
            MXNet NDArray: The sequence broadcast together.
        """
        max_dim = max(map(len, args))
        try:
            assert not [arg for arg in args if 1 < len(arg) < max_dim]
        except AssertionError as error:
            log.error(
                'ERROR: The arguments must be of compatible size: 1 or %i', max_dim
            )
            raise error

        broadcast = [
            arg
            if len(arg) > 1
            else nd.broadcast_axis(arg[0], axis=len(arg.shape) - 1, size=max_dim)
            for arg in args
        ]
        return nd.stack(*broadcast)

    def shape(self, tensor):
        """
        NB: Returns a tuple of longs
        """
        return tensor.shape

    def reshape(self, tensor, newshape):
        return nd.reshape(tensor, newshape)

    def einsum(self, subscripts, *operands):
        """
        A generalized contraction between tensors of arbitrary dimension.

        Warning: not implemented in MXNet

        Args:
            subscripts: str, specifies the subscripts for summation
            operands: list of array_like, these are the tensors for the operation

        Returns:
            tensor: the calculation based on the Einstein summation convention
        """
        raise NotImplementedError("mxnet::einsum is not implemented.")

    def poisson_logpdf(self, n, lam):
        return n * nd.log(lam) - lam - nd.gammaln(n + 1.0)

    def poisson(self, n, lam):
        r"""
        The continous approximation, using :math:`n! = \Gamma\left(n+1\right)`,
        to the probability mass function of the Poisson distribution evaluated
        at :code:`n` given the parameter :code:`lam`.

        Example:

            >>> import pyhf
            >>> pyhf.set_backend(pyhf.tensor.mxnet_backend())
            >>> values = pyhf.tensorlib.astensor([5., 9.])
            >>> rates = pyhf.tensorlib.astensor([6., 8.])
            >>> pyhf.tensorlib.poisson(values, rates)
            <BLANKLINE>
            [0.16062315 0.12407687]
            <NDArray 2 @cpu(0)>


        Args:
            n (Number or Tensor): The value at which to evaluate the approximation to the Poisson distribution p.m.f.
                                  (the observed number of events)
            lam (Number or Tensor): The mean of the Poisson distribution p.d.f.
                                    (the expected number of events)

        Returns:
            MXNet NDArray: Value of the continous approximation to Poisson(n|lam)
        """
        # This is currently copied directly from PyTorch's source until a better
        # way can be found to do this in MXNet
        # https://github.com/pytorch/pytorch/blob/39520ffec15ab7e97691fed048de1832e83785e8/torch/distributions/poisson.py#L59-L63
        return nd.exp((nd.log(lam) * n) - lam - nd.gammaln(n + 1.0))

    def normal_logpdf(self, x, mu, sigma):
        # This is currently copied directly from PyTorch's source until a better
        # way can be found to do this in MXNet
        # https://github.com/pytorch/pytorch/blob/39520ffec15ab7e97691fed048de1832e83785e8/torch/distributions/normal.py#L70-L76
        variance = sigma ** 2
        log_scale = math.log(sigma) if isinstance(sigma, Number) else sigma.log()
        return (
            -((x - mu) ** 2) / (2 * variance)
            - log_scale
            - math.log(math.sqrt(2 * math.pi))
        )

    def normal(self, x, mu, sigma):
        r"""
        The probability density function of the Normal distribution evaluated
        at :code:`x` given parameters of mean of :code:`mu` and standard deviation
        of :code:`sigma`.

        Example:

            >>> import pyhf
            >>> pyhf.set_backend(pyhf.tensor.mxnet_backend())
            >>> values = pyhf.tensorlib.astensor([0.5, 2.0])
            >>> means = pyhf.tensorlib.astensor([0., 2.3])
            >>> sigmas = pyhf.tensorlib.astensor([1., 0.8])
            >>> pyhf.tensorlib.normal(values, means, sigmas)
            <BLANKLINE>
            [0.35206532 0.4648189 ]
            <NDArray 2 @cpu(0)>

        Args:
            x (Number or Tensor): The value at which to evaluate the Normal distribution p.d.f.
            mu (Number or Tensor): The mean of the Normal distribution
            sigma (Number or Tensor): The standard deviation of the Normal distribution

        Returns:
            MXNet NDArray: Value of Normal(x|mu, sigma).
        """
        # This is currently copied directly from PyTorch's source until a better
        # way can be found to do this in MXNet
        # https://github.com/pytorch/pytorch/blob/39520ffec15ab7e97691fed048de1832e83785e8/torch/distributions/normal.py#L70-L76
        return self.exp(self.normal_logpdf(x, mu, sigma))

    def normal_cdf(self, x, mu=0, sigma=1):
        """
        The cumulative distribution function for the Normal distribution

        Example:

            >>> import pyhf
            >>> pyhf.set_backend(pyhf.tensor.mxnet_backend())
            >>> pyhf.tensorlib.normal_cdf(0.8)
            <BLANKLINE>
            [0.7881446]
            <NDArray 1 @cpu(0)>
            >>> values = pyhf.tensorlib.astensor([0.8, 2.0])
            >>> pyhf.tensorlib.normal_cdf(values)
            <BLANKLINE>
            [0.7881446  0.97724986]
            <NDArray 2 @cpu(0)>


        Args:
            x (`tensor` or `float`): The observed value of the random variable to evaluate the CDF for
            mu (`tensor` or `float`): The mean of the Normal distribution
            sigma (`tensor` or `float`): The standard deviation of the Normal distribution

        Returns:
            MXNet NDArray: The CDF
        """
        log.warning(
            'normal_cdf currently uses SciPy stats until pure MXNet distribuiton support is available.'
        )
        x = self.astensor(x).asnumpy()
        mu = self.astensor(mu).asnumpy()
        sigma = self.astensor(sigma).asnumpy()
        return self.astensor(norm.cdf(x, loc=mu, scale=sigma))
