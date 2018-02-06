import numpy as np
import logging
from scipy.special import gammaln
from scipy.stats import norm
log = logging.getLogger(__name__)

class numpy_backend(object):
    def __init__(self, **kwargs):
        self.pois_from_norm = kwargs.get('poisson_from_normal',False)

    def tolist(self,tensor_in):
        tensor_in = self.astensor(tensor_in)
        return tensor_in.tolist()

    def outer(self, tensor_in_1, tensor_in_2):
        tensor_in_1 = self.astensor(tensor_in_1)
        tensor_in_2 = self.astensor(tensor_in_2)
        return np.outer(tensor_in_1,tensor_in_2)

    def astensor(self,tensor_in):
        return np.array(tensor_in)

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
        return np.broadcast_arrays(*args)

    def poisson(self, n, lam):
        n = np.asarray(n)
        if self.pois_from_norm:
            return self.normal(n,lam, self.sqrt(lam))
        return np.exp(n*np.log(lam)-lam-gammaln(n+1.))

    def normal(self, x, mu, sigma):
        return norm.pdf(x, loc=mu, scale=sigma)
