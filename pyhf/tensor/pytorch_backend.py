import torch
import torch.autograd
import logging
log = logging.getLogger(__name__)

class pytorch_backend(object):
    def __init__(self, **kwargs):
        pass

    def clip(self, tensor_in, min, max):
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
            min (`scalar` or `tensor` or `None`): The minimum value to be cliped to
            max (`scalar` or `tensor` or `None`): The maximum value to be cliped to

        Returns:
            PyTorch tensor: A clipped `tensor`
        """
        tensor_in = self.astensor(tensor_in)
        return torch.clamp(tensor_in, min, max)

    def tolist(self, tensor_in):
        tensor_in = self.astensor(tensor_in)
        return tensor_in.data.numpy().tolist()

    def outer(self, tensor_in_1, tensor_in_2):
        tensor_in_1 = self.astensor(tensor_in_1)
        tensor_in_2 = self.astensor(tensor_in_2)
        return torch.ger(tensor_in_1,tensor_in_2)

    def astensor(self, tensor_in):
        """
        Convert to a PyTorch Tensor.

        Args:
            tensor_in (Number or Tensor): Tensor object

        Returns:
            torch.Tensor: A multi-dimensional matrix containing elements of a single data type.
        """
        if isinstance(tensor_in, torch.Tensor):
            v = tensor_in
        else:
            if not isinstance(tensor_in, list):
                tensor_in = [tensor_in]
            v = torch.Tensor(tensor_in)
        return v.type(torch.FloatTensor)

    def sum(self, tensor_in, axis=None):
        tensor_in = self.astensor(tensor_in)
        return torch.sum(tensor_in) if (axis is None or tensor_in.shape == torch.Size([])) else torch.sum(tensor_in, axis)

    def product(self, tensor_in, axis=None):
        tensor_in = self.astensor(tensor_in)
        return torch.prod(tensor_in) if axis is None else torch.prod(tensor_in, axis)

    def ones(self, shape):
        return torch.Tensor(torch.ones(shape))

    def power(self, tensor_in_1, tensor_in_2):
        tensor_in_1 = self.astensor(tensor_in_1)
        tensor_in_2 = self.astensor(tensor_in_2)
        return torch.pow(tensor_in_1, tensor_in_2)

    def sqrt(self,tensor_in):
        tensor_in = self.astensor(tensor_in)
        return torch.sqrt(tensor_in)

    def divide(self,tensor_in_1, tensor_in_2):
        tensor_in_1 = self.astensor(tensor_in_1)
        tensor_in_2 = self.astensor(tensor_in_2)
        return torch.div(tensor_in_1, tensor_in_2)

    def log(self,tensor_in):
        tensor_in = self.astensor(tensor_in)
        return torch.log(tensor_in)

    def exp(self,tensor_in):
        tensor_in = self.astensor(tensor_in)
        return torch.exp(tensor_in)

    def stack(self, sequence, axis = 0):
        return torch.stack(sequence,dim = axis)

    def where(self, mask, tensor_in_1, tensor_in_2):
        mask = self.astensor(mask)
        tensor_in_1 = self.astensor(tensor_in_1)
        tensor_in_2 = self.astensor(tensor_in_2)
        return mask * tensor_in_1 + (1-mask) * tensor_in_2

    def concatenate(self, sequence):
        return torch.cat(sequence)

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
            log.error('ERROR: The arguments must be of compatible size: 1 or %i', max_dim)
            raise error

        broadcast = [arg if len(arg) > 1 else arg.expand(max_dim)
                     for arg in args]
        return broadcast

    def poisson(self, n, lam):
        return self.normal(n,lam, self.sqrt(lam))

    def normal(self, x, mu, sigma):
        x = self.astensor(x)
        mu = self.astensor(mu)
        sigma = self.astensor(sigma)
        normal = torch.distributions.Normal(mu, sigma)
        return self.exp(normal.log_prob(x))

    def normal_cdf(self, x, mu=[0.], sigma=[1.]):
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
