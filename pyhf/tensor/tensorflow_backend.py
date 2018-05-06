import tensorflow as tf
import logging
log = logging.getLogger(__name__)

class tensorflow_backend(object):
    def __init__(self, **kwargs):
        self.session = kwargs.get('session')

    def clip(self, tensor_in, min, max):
        """
        Clips (limits) the tensor values to be within a specified min and max.

        Example:

            >>> import pyhf, tensorflow
            >>> sess = tensorflow.Session()
            ...
            >>> pyhf.set_backend(pyhf.tensor.tensorflow_backend())
            >>> a = pyhf.tensorlib.astensor([-2, -1, 0, 1, 2])
            >>> with sess.as_default():
            ...   pyhf.tensorlib.clip(a, -1, 1).eval()
            ...
            array([-1., -1.,  0.,  1.,  1.], dtype=float32)

        Args:
            tensor_in (`tensor`): The input tensor object
            min (`scalar` or `tensor` or `None`): The minimum value to be cliped to
            max (`scalar` or `tensor` or `None`): The maximum value to be cliped to

        Returns:
            TensorFlow Tensor: A clipped `tensor`
        """
        tensor_in = self.astensor(tensor_in)
        if min is None:
            min = tf.reduce_min(tensor_in)
        if max is None:
            max = tf.reduce_max(tensor_in)
        return tf.clip_by_value(tensor_in, min, max)

    def tolist(self,tensor_in):
        tensor_in = self.astensor(tensor_in)
        return self.session.run(tensor_in).tolist()

    def outer(self, tensor_in_1, tensor_in_2):
        tensor_in_1 = self.astensor(tensor_in_1)
        tensor_in_2 = self.astensor(tensor_in_2)
        tensor_in_1 = tensor_in_1 if tensor_in_1.dtype is not tf.bool else tf.cast(tensor_in_1, tf.float32)
        tensor_in_1 = tensor_in_1 if tensor_in_2.dtype is not tf.bool else tf.cast(tensor_in_2, tf.float32)
        return tf.einsum('i,j->ij', tensor_in_1, tensor_in_2)

    def astensor(self, tensor_in, dtype=tf.float32):
        if isinstance(tensor_in, tf.Tensor):
            v = tensor_in
        else:
            v = tf.convert_to_tensor(tensor_in)
        if v.dtype is not dtype:
          v = tf.cast(v, dtype)
        return v

    def sum(self, tensor_in, axis=None):
        tensor_in = self.astensor(tensor_in)
        return tf.reduce_sum(tensor_in) if (axis is None or tensor_in.shape == tf.TensorShape([])) else tf.reduce_sum(tensor_in, axis)

    def product(self, tensor_in, axis=None):
        tensor_in = self.astensor(tensor_in)
        return tf.reduce_prod(tensor_in) if axis is None else tf.reduce_prod(tensor_in, axis)

    def ones(self, shape):
        return tf.ones(shape)

    def power(self,tensor_in_1, tensor_in_2):
        tensor_in_1 = self.astensor(tensor_in_1)
        tensor_in_2 = self.astensor(tensor_in_2)
        return tf.pow(tensor_in_1, tensor_in_2)

    def sqrt(self,tensor_in):
        tensor_in = self.astensor(tensor_in)
        return tf.sqrt(tensor_in)

    def divide(self,tensor_in_1, tensor_in_2):
        tensor_in_1 = self.astensor(tensor_in_1)
        tensor_in_2 = self.astensor(tensor_in_2)
        return tf.divide(tensor_in_1, tensor_in_2)

    def log(self,tensor_in):
        tensor_in = self.astensor(tensor_in)
        return tf.log(tensor_in)

    def exp(self,tensor_in):
        tensor_in = self.astensor(tensor_in)
        return tf.exp(tensor_in)

    def stack(self, sequence, axis = 0):
        return tf.stack(sequence, axis = axis)

    def where(self, mask, tensor_in_1, tensor_in_2):
        mask = self.astensor(mask)
        tensor_in_1 = self.astensor(tensor_in_1)
        tensor_in_2 = self.astensor(tensor_in_2)
        return mask * tensor_in_1 + (1-mask) * tensor_in_2

    def concatenate(self, sequence):
        return tf.concat(sequence, axis=0)

    def simple_broadcast(self, *args):
        broadcast = []
        def generic_len(a):
          try:
            return len(a)
          except TypeError:
            if len(a.shape) < 1:
              return 0
            else:
              return a.shape[0]

        maxdim = max(map(generic_len,args))
        for a in args:
            broadcast.append(self.astensor(a) if generic_len(a) > 1 else a*self.ones(maxdim))
        return broadcast

    def poisson(self, n, lam):
        # could be changed to actual Poisson easily
        return self.normal(n,lam, self.sqrt(lam))

    def normal(self, x, mu, sigma):
        x = self.astensor(x)
        mu = self.astensor(mu)
        sigma = self.astensor(sigma)
        normal = tf.distributions.Normal(mu, sigma)
        return normal.prob(x)

    def normal_cdf(self, x, mu=0, sigma=1):
        """
        The cumulative distribution function for the Normal distribution

        Example:

            >>> import pyhf, tensorflow
            >>> sess = tensorflow.Session()
            ...
            >>> pyhf.set_backend(pyhf.tensor.tensorflow_backend())
            >>> with sess.as_default():
            ...   pyhf.tensorlib.normal_cdf(0.8).eval()
            ...
            0.7881446

        Args:
            x (`tensor` or `float`): The observed value of the random variable to evaluate the CDF for
            mu (`tensor` or `float`): The mean of the Normal distribution
            sigma (`tensor` or `float`): The standard deviation of the Normal distribution

        Returns:
            TensorFlow Tensor: The CDF
        """
        x = self.astensor(x)
        mu = self.astensor(mu)
        sigma = self.astensor(sigma)
        normal = tf.distributions.Normal(mu, sigma)
        return normal.cdf(x)
