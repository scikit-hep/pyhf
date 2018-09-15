import dask.array as da
import logging
# we can use these scipy methods because of
# generalized ufuncs (might not work for delayed
# dask.. recheck when we get there)
# http://dask.pydata.org/en/latest/array-gufunc.html
from scipy.special import gammaln
from scipy.stats import norm

log = logging.getLogger(__name__)


class dask_backend(object):

    def __init__(self, **kwargs):
        self.pois_from_norm = kwargs.get('poisson_from_normal', False)

    def clip(self, tensor_in, min, max):
        """
        Clips (limits) the tensor values to be within a specified min and max.

        Example:

            >>> import pyhf
            >>> pyhf.set_backend(pyhf.tensor.dask_backend())
            >>> a = pyhf.tensorlib.astensor([-2, -1, 0, 1, 2])
            >>> pyhf.tensorlib.clip(a, -1, 1).compute()
            array([-1, -1,  0,  1,  1])

        Args:
            tensor_in (`tensor`): The input tensor object
            min (`scalar` or `tensor` or `None`): The minimum value to be cliped to
            max (`scalar` or `tensor` or `None`): The maximum value to be cliped to

        Returns:
            `dask.array`: A clipped `tensor`
        """
        tensor_in = self.astensor(tensor_in)
        return da.clip(tensor_in, min, max)

    def tolist(self, tensor_in):
        tensor_in = self.astensor(tensor_in)
        return tensor_in.compute().tolist()

    def outer(self, tensor_in_1, tensor_in_2):
        tensor_in_1 = self.astensor(tensor_in_1)
        tensor_in_2 = self.astensor(tensor_in_2)
        return da.outer(tensor_in_1, tensor_in_2)

    def astensor(self, tensor_in):
        """
        Convert to a NumPy array.

        Args:
            tensor_in (Number or Tensor): Tensor object

        Returns:
            `dask.array`: A multi-dimensional, fixed-size homogenous array.
        """
        return da.asarray(tensor_in)

    def sum(self, tensor_in, axis=None):
        tensor_in = self.astensor(tensor_in)
        return da.sum(tensor_in, axis=axis)

    def product(self, tensor_in, axis=None):
        tensor_in = self.astensor(tensor_in)
        return da.prod(tensor_in, axis=axis)

    def abs(self, tensor):
        """
        Calculate the absolute value element-wise.

        Args:
            tensor (Number or Tensor): Tensor object

        Returns:
            `dask.array`: The absolute value of the tensor element-wise.
        """
        tensor = self.astensor(tensor)
        return da.fabs(tensor)

    def ones(self, shape):
        return da.ones(shape, chunks=100)

    def zeros(self, shape):
        return da.zeros(shape, chunks=100)

    def power(self, tensor_in_1, tensor_in_2):
        tensor_in_1 = self.astensor(tensor_in_1)
        tensor_in_2 = self.astensor(tensor_in_2)
        return da.power(tensor_in_1, tensor_in_2)

    def sqrt(self, tensor_in):
        tensor_in = self.astensor(tensor_in)
        return da.sqrt(tensor_in)

    def divide(self, tensor_in_1, tensor_in_2):
        tensor_in_1 = self.astensor(tensor_in_1)
        tensor_in_2 = self.astensor(tensor_in_2)
        return da.divide(tensor_in_1, tensor_in_2)

    def log(self, tensor_in):
        tensor_in = self.astensor(tensor_in)
        return da.log(tensor_in)

    def exp(self, tensor_in):
        tensor_in = self.astensor(tensor_in)
        return da.exp(tensor_in)

    def stack(self, sequence, axis=0):
        return da.stack(sequence, axis=axis)

    def where(self, mask, tensor_in_1, tensor_in_2):
        mask = self.astensor(mask)
        tensor_in_1 = self.astensor(tensor_in_1)
        tensor_in_2 = self.astensor(tensor_in_2)
        return da.where(mask, tensor_in_1, tensor_in_2)

    def concatenate(self, sequence, axis=0):
        """
        Join a sequence of arrays along an existing axis.

        Args:
            sequence: sequence of tensors
            axis: dimension along which to concatenate

        Returns:
            `dask.array`: The concatenated tensor.
         """
        return da.concatenate(sequence, axis=axis)

    def simple_broadcast(self, *args):
        """
        Broadcast a sequence of 1 dimensional arrays.

        Example:

            >>> import pyhf
            >>> pyhf.set_backend(pyhf.tensor.dask_backend())
            >>> [x.compute() for x in pyhf.tensorlib.simple_broadcast(
            ...   pyhf.tensorlib.astensor([1]),
            ...   pyhf.tensorlib.astensor([2, 3, 4]),
            ...   pyhf.tensorlib.astensor([5, 6, 7]))]
            [array([1, 1, 1]), array([2, 3, 4]), array([5, 6, 7])]

        Args:
            args (Array of Tensors): Sequence of arrays

        Returns:
            list of Tensors: The sequence broadcast together.
        """
        return da.broadcast_arrays(*args)

    def einsum(self, subscripts, *operands):
        """
        Evaluates the Einstein summation convention on the operands.

        Using the Einstein summation convention, many common multi-dimensional
        array operations can be represented in a simple fashion. This function
        provides a way to compute such summations. The best way to understand
        this function is to try the examples below, which show how many common
        NumPy functions can be implemented as calls to einsum.

        Args:
            subscripts (`str`): Specifies the subscripts for summation
            operands (list of array_like): The tensors for the operation

        Returns:
            `dask.array`: The calculation based on the Einstein summation convention
        """
        return da.einsum(subscripts, *operands)

    def poisson(self, n, lam):
        n = da.asarray(n)
        if self.pois_from_norm:
            return self.normal(n, lam, self.sqrt(lam))
        return da.exp(n * da.log(lam) - lam - gammaln(n + 1.))

    def normal(self, x, mu, sigma):
        # works because of generalized ufuncs
        return norm.pdf(x, loc=mu, scale=sigma)

    def normal_cdf(self, x, mu=0, sigma=1):
        """
        The cumulative distribution function for the Normal distribution

        Example:

            >>> import pyhf
            >>> pyhf.set_backend(pyhf.tensor.dask_backend())
            >>> pyhf.tensorlib.normal_cdf(0.8)
            0.7881446014166034

        Args:
            x (`tensor` or `float`): The observed value of the random variable to evaluate the CDF for
            mu (`tensor` or `float`): The mean of the Normal distribution
            sigma (`tensor` or `float`): The standard deviation of the Normal distribution

        Returns:
            NumPy float: The CDF
        """
        return norm.cdf(x, loc=mu, scale=sigma)  # works because of generalized ufuncs
