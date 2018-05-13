from mxnet import nd
import logging
import math  # Required for normal()
from numbers import Number  # Required for normal()
from scipy.stats import norm  # Required for normal_cdf()
log = logging.getLogger(__name__)


class mxnet_backend(object):
    """MXNet backend for pyhf"""

    def __init__(self, **kwargs):
        pass

    def clip(self, tensor_in, min, max):
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
            min (`scalar` or `tensor` or `None`): The minimum value to be cliped to
            max (`scalar` or `tensor` or `None`): The maximum value to be cliped to

        Returns:
            MXNet NDArray: A clipped `tensor`
        """
        tensor_in = self.astensor(tensor_in)
        return nd.clip(tensor_in, min, max)

    def tolist(self, tensor_in):
        """
        Convert a tensor to a list.

        Args:
            tensor_in (Tensor): Input MXNet tensor

        Returns:
            list: The possibly nested list of tensor elements.
        """
        tensor_in = self.astensor(tensor_in)
        return tensor_in.asnumpy().tolist()

    def outer(self, tensor_in_1, tensor_in_2):
        """
        The outer product of two tensors.

        Args:
            tensor_in_1 (Tensor): Tensor object
            tensor_in_2 (Tensor): Tensor object

        Returns:
            MXNet NDArray: The outer product.
        """
        tensor_in_1 = self.astensor(tensor_in_1)
        tensor_in_2 = self.astensor(tensor_in_2)

        tensor_1_shape = tensor_in_1.shape
        tensor_2_shape = tensor_in_2.shape
        if len(tensor_1_shape) == 1:
            tensor_1_shape = (tensor_1_shape[0], 1)
        if len(tensor_2_shape) == 1:
            tensor_2_shape = (tensor_2_shape[0], 1)

        rows1, cols1 = tensor_1_shape
        rows2, cols2 = tensor_2_shape
        return nd.reshape(nd.broadcast_mul(tensor_in_1.reshape((rows1, 1, cols1, 1)),
                                           tensor_in_2.reshape((1, rows2, 1, cols2))),
                          (rows1 * cols1, rows2 * cols2))

    def astensor(self, tensor_in):
        """
        Convert to a MXNet NDArray.

        Args:
            tensor_in (Number or Tensor): Tensor object

        Returns:
            MXNet NDArray: A multi-dimensional, fixed-size homogenous array.
        """
        try:
            tensor = nd.array(tensor_in)
        except ValueError:
            tensor = nd.array([tensor_in])
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
        tensor_in = self.astensor(tensor_in)
        if axis is None or tensor_in.shape == nd.array([]).size:
            return nd.sum(tensor_in)
        else:
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
        tensor_in = self.astensor(tensor_in)
        if axis is None:
            return nd.prod(tensor_in)
        else:
            return nd.prod(tensor_in, axis)

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
        tensor_in_1 = self.astensor(tensor_in_1)
        tensor_in_2 = self.astensor(tensor_in_2)
        return nd.power(tensor_in_1, tensor_in_2)

    def sqrt(self, tensor_in):
        """
        Element-wise square-root value of the input.

        Args:
            tensor_in (Tensor): Tensor object

        Returns:
            MXNet NDArray: Element-wise square-root value.
        """
        tensor_in = self.astensor(tensor_in)
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
        tensor_in_1 = self.astensor(tensor_in_1)
        tensor_in_2 = self.astensor(tensor_in_2)
        return nd.divide(tensor_in_1, tensor_in_2)

    def log(self, tensor_in):
        """
        Element-wise Natural logarithmic value of the input.

        Args:
            tensor_in (Tensor): Tensor object

        Returns:
            MXNet NDArray: Element-wise Natural logarithmic value.
        """
        tensor_in = self.astensor(tensor_in)
        return nd.log(tensor_in)

    def exp(self, tensor_in):
        """
        Element-wise exponential value of the input.

        Args:
            tensor_in (Tensor): Tensor object

        Returns:
            MXNet NDArray: Element-wise exponential value.
        """
        tensor_in = self.astensor(tensor_in)
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
        mask = self.astensor(mask)
        tensor_in_1 = self.astensor(tensor_in_1)
        tensor_in_2 = self.astensor(tensor_in_2)
        return nd.add(nd.multiply(mask, tensor_in_1),
                      nd.multiply(nd.subtract(1, mask), tensor_in_2))

    def concatenate(self, sequence):
        """
        Join the elements of the sequence.

        Args:
            sequence (Array of Tensors): The sequence of arrays to join

        Returns:
            MXNet NDArray: The ndarray of the joined elements.
        """
        return nd.concat(*sequence, dim=0)

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
        args = [self.astensor(arg) for arg in args]
        max_dim = max(map(len, args))
        try:
            assert len([arg for arg in args if 1 < len(arg) < max_dim]) == 0
        except AssertionError as error:
            log.error(
                'ERROR: The arguments must be of compatible size: 1 or %i', max_dim)
            raise error

        broadcast = [arg if len(arg) > 1
                     else nd.broadcast_axis(arg[0], axis=len(arg.shape) - 1, size=max_dim)
                     for arg in args]
        return nd.stack(*broadcast)

    def poisson(self, n, lam):
        """
        The continous approximation to the probability density function of the Poisson
        distribution given the parameters evaluated at `n`.

        Args:
            n (Number or Tensor): The value at which to evaluate the Poisson distribution p.d.f.
                                  (the observed number of events)
            lam (Number or Tensor): The mean of the Poisson distribution p.d.f.
                                    (the expected number of events)

        Returns:
            MXNet NDArray: Value of N(n|lam, sqrt(lam)), the continous approximation to Poisson(n|lam).
        """
        return self.normal(n, lam, self.sqrt(lam))

    def normal(self, x, mu, sigma):
        """
        The probability density function of the Normal distribution given the parameters
        evaluated at `x`.

        Args:
            x (Number or Tensor): The point at which to evaluate the Normal distribution p.d.f.
            mu (Number or Tensor): The mean of the Normal distribution p.d.f.
            sigma(Number or Tensor): The standard deviation of the Normal distribution p.d.f.

        Returns:
            MXNet NDArray: Value of N(x|mu, sigma).
        """
        x = self.astensor(x)
        mu = self.astensor(mu)
        sigma = self.astensor(sigma)

        # This is currently copied directly from PyTorch's source until a better
        # way can be found to do this in MXNet
        # https://github.com/pytorch/pytorch/blob/master/torch/distributions/normal.py#L61-L66
        def log_prob(value, loc, scale):
            variance = scale ** 2
            log_scale = math.log(scale) if isinstance(
                scale, Number) else scale.log()
            return -((value - loc) ** 2) / (2 * variance) - log_scale - math.log(math.sqrt(2 * math.pi))
        return self.exp(log_prob(x, mu, sigma))

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

        Args:
            x (`tensor` or `float`): The observed value of the random variable to evaluate the CDF for
            mu (`tensor` or `float`): The mean of the Normal distribution
            sigma (`tensor` or `float`): The standard deviation of the Normal distribution

        Returns:
            MXNet NDArray: The CDF
        """
        log.warning(
            'normal_cdf currently uses SciPy stats until pure MXNet distribuiton support is available.')
        x = self.astensor(x).asnumpy()
        mu = self.astensor(mu).asnumpy()
        sigma = self.astensor(sigma).asnumpy()
        return self.astensor(norm.cdf(x, loc=mu, scale=sigma))
