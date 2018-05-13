import numpy as np
import logging
from scipy.special import gammaln
from scipy.stats import norm
log = logging.getLogger(__name__)

class numpy_backend(object):
    def __init__(self, **kwargs):
        self.pois_from_norm = kwargs.get('poisson_from_normal',False)

    def clip(self, tensor_in, min, max):
        """
        Clips (limits) the tensor values to be within a specified min and max.

        Example:

            >>> import pyhf
            >>> pyhf.set_backend(pyhf.tensor.numpy_backend())
            >>> a = pyhf.tensorlib.astensor([-2, -1, 0, 1, 2])
            >>> pyhf.tensorlib.clip(a, -1, 1)
            array([-1, -1,  0,  1,  1])

        Args:
            tensor_in (`tensor`): The input tensor object
            min (`scalar` or `tensor` or `None`): The minimum value to be cliped to
            max (`scalar` or `tensor` or `None`): The maximum value to be cliped to

        Returns:
            NumPy ndarray: A clipped `tensor`
        """
        tensor_in = self.astensor(tensor_in)
        return np.clip(tensor_in, min, max)

    def tolist(self,tensor_in):
        tensor_in = self.astensor(tensor_in)
        return tensor_in.tolist()

    def outer(self, tensor_in_1, tensor_in_2):
        tensor_in_1 = self.astensor(tensor_in_1)
        tensor_in_2 = self.astensor(tensor_in_2)
        return np.outer(tensor_in_1,tensor_in_2)

    def astensor(self, tensor_in):
        """
        Convert to a NumPy array.

        Args:
            tensor_in (Number or Tensor): Tensor object

        Returns:
            `numpy.ndarray`: A multi-dimensional, fixed-size homogenous array.
        """
        return np.asarray(tensor_in)

    def sum(self, tensor_in, axis=None):
        tensor_in = self.astensor(tensor_in)
        return np.sum(tensor_in, axis=axis)

    def product(self, tensor_in, axis=None):
        tensor_in = self.astensor(tensor_in)
        return np.product(tensor_in, axis = axis)

    def ones(self,shape):
        return np.ones(shape)

    def power(self,tensor_in_1, tensor_in_2):
        tensor_in_1 = self.astensor(tensor_in_1)
        tensor_in_2 = self.astensor(tensor_in_2)
        return np.power(tensor_in_1, tensor_in_2)

    def sqrt(self,tensor_in):
        tensor_in = self.astensor(tensor_in)
        return np.sqrt(tensor_in)

    def divide(self,tensor_in_1, tensor_in_2):
        tensor_in_1 = self.astensor(tensor_in_1)
        tensor_in_2 = self.astensor(tensor_in_2)
        return np.divide(tensor_in_1, tensor_in_2)

    def log(self,tensor_in):
        tensor_in = self.astensor(tensor_in)
        return np.log(tensor_in)

    def exp(self,tensor_in):
        tensor_in = self.astensor(tensor_in)
        return np.exp(tensor_in)

    def stack(self, sequence, axis = 0):
        return np.stack(sequence,axis = axis)

    def where(self, mask, tensor_in_1, tensor_in_2):
        mask = self.astensor(mask)
        tensor_in_1 = self.astensor(tensor_in_1)
        tensor_in_2 = self.astensor(tensor_in_2)
        return np.where(mask, tensor_in_1, tensor_in_2)

    def concatenate(self, sequence):
        return np.concatenate(sequence)

    def simple_broadcast(self, *args):
        """
        Broadcast a sequence of 1 dimensional arrays.

        Example:

            >>> import pyhf
            >>> pyhf.set_backend(pyhf.tensor.numpy_backend())
            >>> pyhf.tensorlib.simple_broadcast(
            ...   pyhf.tensorlib.astensor([1]),
            ...   pyhf.tensorlib.astensor([2, 3, 4]),
            ...   pyhf.tensorlib.astensor([5, 6, 7]))
            [array([1, 1, 1]), array([2, 3, 4]), array([5, 6, 7])]

        Args:
            args (Array of Tensors): Sequence of arrays

        Returns:
            list of Tensors: The sequence broadcast together.
        """
        return np.broadcast_arrays(*args)

    def poisson(self, n, lam):
        n = np.asarray(n)
        if self.pois_from_norm:
            return self.normal(n,lam, self.sqrt(lam))
        return np.exp(n*np.log(lam)-lam-gammaln(n+1.))

    def normal(self, x, mu, sigma):
        return norm.pdf(x, loc=mu, scale=sigma)

    def normal_cdf(self, x, mu=0, sigma=1):
        """
        The cumulative distribution function for the Normal distribution

        Example:

            >>> import pyhf
            >>> pyhf.set_backend(pyhf.tensor.numpy_backend())
            >>> pyhf.tensorlib.normal_cdf(0.8)
            0.7881446014166034

        Args:
            x (`tensor` or `float`): The observed value of the random variable to evaluate the CDF for
            mu (`tensor` or `float`): The mean of the Normal distribution
            sigma (`tensor` or `float`): The standard deviation of the Normal distribution

        Returns:
            NumPy float: The CDF
        """
        return norm.cdf(x, loc=mu, scale=sigma)
