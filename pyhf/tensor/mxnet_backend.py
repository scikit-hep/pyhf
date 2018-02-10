import mxnet as mx
from mxnet import nd
import logging
log = logging.getLogger(__name__)


class mxnet_backend(object):
    """Backend for MXNet"""

    def __init__(self, **kwargs):
        self.session = kwargs.get('session')

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
        The outer product of two tensors: u.v^T

        Args:
            tensor_in_1: tensor object
            tensor_in_2: tensor object

        Returns:
            MXNet NDArray: The outer product
        """
        tensor_in_1 = self.astensor(tensor_in_1)
        tensor_in_2 = self.astensor(tensor_in_2)
        pass

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
        if axis is None or tensor_in.shape == nd.Size([]):
            return nd.sum(tensor_in)
        else:
            return nd.sum(tensor_in, axis)

    def product(self, tensor_in, axis=None):
        pass

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
        pass

    def where(self, mask, tensor_in_1, tensor_in_2):
        pass

    def concatenate(self, sequence):
        pass

    def simple_broadcast(self, *args):
        pass

    def poisson(self, n, lam):
        pass

    def normal(self, x, mu, sigma):
        pass
