import torch
import torch.autograd
import logging
log = logging.getLogger(__name__)

class pytorch_backend(object):
    def __init__(self, **kwargs):
        pass

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
                v = torch.autograd.Variable(tensor_in)
        return v.type(torch.FloatTensor)

    def sum(self, tensor_in, axis=None):
        tensor_in = self.astensor(tensor_in)
        return torch.sum(tensor_in) if (axis is None or  tensor_in.shape == torch.Size([])) else torch.sum(tensor_in, axis)

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
        normal = torch.distributions.Normal(mu,sigma)
        return self.exp(normal.log_prob(x))
