from mxnet import nd
import logging
log = logging.getLogger(__name__)


class mxnet_backend(object):
    """Backend for MXNet"""

    def __init__(self, **kwargs):
        pass

    def tolist(self, tensor_in):
        """
        Convert a tensor to a list

        Args:
            tensor_in: MXNet tensor

        Returns:
            The possibly nested list of tensor elements.
        """
        tensor_in = self.astensor(tensor_in)
        return tensor_in.asnumpy().tolist()

    def outer(self, tensor_in_1, tensor_in_2):
        """
        The outer product of two tensors

        Args:
            tensor_in_1: tensor object
            tensor_in_2: tensor object

        Returns:
            MXNet NDArray: The outer product
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
        return nd.reshape(nd.dot(tensor_in_1.reshape((rows1, 1, cols1, 1)),
                                 tensor_in_2.reshape((1, rows2, 1, cols2))),
                          (rows1 * cols1, rows2 * cols2))

    def astensor(self, tensor_in):
        """
        Convert a tensor to an MXNet NDArray

        Args:
            tensor_in: tensor object

        Returns:
            MXNet NDArray: a multi-dimensional, fixed-size homogenous array.
        """
        return nd.array(tensor_in)

    def sum(self, tensor_in, axis=None):
        """
        Compute the sum of array elements over given axes.

        Args:
            tensor_in: tensor object
            axis: the axes over which to sum

        Returns:
            MXNet NDArray: ndarray of the sum over the axes
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
            tensor_in: tensor object
            axis: the axes over which to take the product

        Returns:
            MXNet NDArray: ndarray of the product over the axes
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
            shape: the shape of the array

        Returns:
            MXNet NDArray: ndarray of 1's with given shape
        """
        return nd.ones(shape)

    def zeros(self, shape):
        """
        A new array filled with all zeros, with the given shape.

        Args:
            shape: the shape of the array

        Returns:
            MXNet NDArray: ndarray of 0's with given shape
        """
        return nd.zeros(shape)

    def power(self, tensor_in_1, tensor_in_2):
        """
        Result of first array elements raised to powers from second array,
        element-wise with broadcasting.

        Args:
            tensor_in_1: tensor object
            tensor_in_2: tensor object

        Returns:
            MXNet NDArray: first array elements raised to powers from second array
        """
        tensor_in_1 = self.astensor(tensor_in_1)
        tensor_in_2 = self.astensor(tensor_in_2)
        return nd.power(tensor_in_1, tensor_in_2)

    def sqrt(self, tensor_in):
        """
        Element-wise square-root value of the input.

        Args:
            tensor_in: tensor object

        Returns:
            MXNet NDArray: element-wise square-root value
        """
        tensor_in = self.astensor(tensor_in)
        return nd.sqrt(tensor_in)

    def divide(self, tensor_in_1, tensor_in_2):
        """
        Element-wise division of the input arrays with broadcasting.

        Args:
            tensor_in_1: tensor object
            tensor_in_2: tensor object

        Returns:
            MXNet NDArray: element-wise division of the input arrays
        """
        tensor_in_1 = self.astensor(tensor_in_1)
        tensor_in_2 = self.astensor(tensor_in_2)
        return nd.divide(tensor_in_1, tensor_in_2)

    def log(self, tensor_in):
        """
        Element-wise Natural logarithmic value of the input.

        Args:
            tensor_in: tensor object

        Returns:
            MXNet NDArray: element-wise Natural logarithmic value
        """
        tensor_in = self.astensor(tensor_in)
        return nd.log(tensor_in)

    def exp(self, tensor_in):
        """
        Element-wise exponential value of the input.

        Args:
            tensor_in: tensor object

        Returns:
            MXNet NDArray: element-wise exponential value
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
            sequence: sequence of arrays
            axis: the axis along which to join the arrays

        Returns:
            MXNet NDArray: ndarray comprised of the elements of the sequence
        """
        return nd.stack(*sequence, axis=axis)

    def where(self, mask, tensor_in_1, tensor_in_2):
        """
        Apply a boolean selection mask to the elements of the input tensors

        Example:
            >>> mxnet_backend.where(
                mxnet_backend.astensor([1, 0, 1]),
                mxnet_backend.astensor([1, 1, 1]),
                mxnet_backend.astensor([2, 2, 2]))
            >>> [1. 2. 1.]

        Args:
            mask: Boolean mask (boolean or tensor object of booleans)
            tensor_in_1: tensor object
            tensor_in_2: tensor object

        Returns:
            MXNet NDArray: The result of the mask being applied to the tensors
        """
        mask = self.astensor(mask)
        tensor_in_1 = self.astensor(tensor_in_1)
        tensor_in_2 = self.astensor(tensor_in_2)
        return nd.multiply(mask, tensor_in_1) + nd.multiply(nd.subtract(1, mask), tensor_in_2)

    def concatenate(self, sequence):
        """
        Join the elements of the sequence

        Args:
            sequence: the sequence of arrays to join

        Returns:
            MXNet NDArray: The ndarray of the joined elements
        """
        return nd.concat(*sequence, dim=0)

    def simple_broadcast(self, *args):
        """
        There should be a more MXNet-style way to do this
        """
        broadcast = []
        max_dim = max(map(len, args))
        for arg in args:
            broadcast.append(self.astensor(arg)
                             if len(arg) > 1 else arg * self.ones(max_dim))
        return broadcast

    def poisson(self, n, lam):
        return self.normal(n, lam, self.sqrt(lam))

    def normal(self, x, mu, sigma):
        """
        Currently copying from PyTorch's source until can find a better way to do this
        """
        import math
        from numbers import Number
        x = self.astensor(x)
        mu = self.astensor(mu)
        sigma = self.astensor(sigma)

        def log_prob(value, loc, scale):
            variance = scale ** 2
            log_scale = math.log(scale) if isinstance(
                scale, Number) else scale.log()
            return -((value - loc) ** 2) / (2 * variance) - log_scale - math.log(math.sqrt(2 * math.pi))
        return self.exp(log_prob(x, mu, sigma))
