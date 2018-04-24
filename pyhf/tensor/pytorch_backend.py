import torch
import torch.autograd
import math  # Needed only for temp version of normal_cdf
import logging
log = logging.getLogger(__name__)

class pytorch_backend(object):
    def __init__(self, **kwargs):
        pass

    def clip(self, tensor_in, min, max):
        """
        Clips (limits) the tensor values to be within a specified min and max.

        Example::

            >>> a = pyhf.tensorlib.astensor([-2, -1, 0, 1, 2])
            >>> pyhf.tensorlib.clip(a, -1, 1)
            array([-1, -1,  0,  1,  1])

        Args:
            tensor_in (`tensor`): The input tensor object
            min (`scalar` or `tensor` or `None`): The minimum value to be cliped to
            max (`scalar` or `tensor` or `None`): The maximum value to be cliped to

        Returns:
            PyTorch tensor: A clipped `tensor`
        """
        tensor_in = self.astensor(tensor_in)
        return torch.clamp(tensor_in, min, max)

    def tolist(self,tensor_in):
        tensor_in = self.astensor(tensor_in)
        return tensor_in.data.numpy().tolist()

    def outer(self, tensor_in_1, tensor_in_2):
        tensor_in_1 = self.astensor(tensor_in_1)
        tensor_in_2 = self.astensor(tensor_in_2)
        return torch.ger(tensor_in_1,tensor_in_2)

    def astensor(self, tensor_in):
        if isinstance(tensor_in, torch.autograd.Variable):
            v = tensor_in
        else:
            try:
                v = torch.autograd.Variable(torch.Tensor(tensor_in))
            except TypeError:
                try:
                    v = torch.autograd.Variable(tensor_in)
                except RuntimeError:
                    # Guard against a float being passed in and casuing a RuntimeError
                    v = torch.autograd.Variable(torch.Tensor([tensor_in]))
        return v.type(torch.FloatTensor)

    def sum(self, tensor_in, axis=None):
        tensor_in = self.astensor(tensor_in)
        return torch.sum(tensor_in) if (axis is None or tensor_in.shape == torch.Size([])) else torch.sum(tensor_in, axis)

    def product(self, tensor_in, axis=None):
        tensor_in = self.astensor(tensor_in)
        return torch.prod(tensor_in) if axis is None else torch.prod(tensor_in, axis)

    def ones(self, shape):
        return torch.autograd.Variable(torch.ones(shape))

    def power(self,tensor_in_1, tensor_in_2):
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
        broadcast = []
        maxdim = max(map(len,args))
        for a in args:
            broadcast.append(self.astensor(a) if len(a) > 1 else a*self.ones(maxdim))
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

        Example::

            >>> pyhf.tensorlib.normal_cdf([0.8])
            Variable containing:
             0.7881
            [torch.FloatTensor of size 1]

        Args:
            x (`tensor` or `float`): The observed value of the random variable
                                      to evaluate the CDF for
            mu (`tensor` or `float`): The mean of the Normal distribution
            sigma (`tensor` or `float`): The standard deviation of the Normal distribution

        Returns:
            PyTorch FloatTensor: The CDF
        """
        x = self.astensor(x)
        mu = self.astensor(mu)
        sigma = self.astensor(sigma)
        # Normal.cdf() is not yet in latest stable release: 0.3.1
        # c.f. http://pytorch.org/docs/master/_modules/torch/distributions/normal.html#Normal.cdf
        # normal = torch.distributions.Normal(mu, sigma)
        # return normal.cdf(x)
        return 0.5 * (1 + torch.erf((x - mu) * sigma.reciprocal() / math.sqrt(2)))
