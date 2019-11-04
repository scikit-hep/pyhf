import logging
import tensorflow as tf
import tensorflow_probability as tfp

log = logging.getLogger(__name__)


class tensorflow_backend(object):
    """TensorFlow backend for pyhf"""

    def __init__(self, **kwargs):
        self.session = kwargs.get('session')
        self.name = 'tensorflow'

    def clip(self, tensor_in, min_value, max_value):
        """
        Clips (limits) the tensor values to be within a specified min and max.

        Example:

            >>> import pyhf
            >>> import tensorflow as tf
            >>> sess = tf.Session()
            ...
            >>> pyhf.set_backend(pyhf.tensor.tensorflow_backend(session=sess))
            >>> a = pyhf.tensorlib.astensor([-2, -1, 0, 1, 2])
            >>> with sess.as_default():
            ...   sess.run(pyhf.tensorlib.clip(a, -1, 1))
            ...
            array([-1., -1.,  0.,  1.,  1.], dtype=float32)

        Args:
            tensor_in (`tensor`): The input tensor object
            min_value (`scalar` or `tensor` or `None`): The minimum value to be cliped to
            max_value (`scalar` or `tensor` or `None`): The maximum value to be cliped to

        Returns:
            TensorFlow Tensor: A clipped `tensor`
        """
        if min_value is None:
            min_value = tf.reduce_min(tensor_in)
        if max_value is None:
            max_value = tf.reduce_max(tensor_in)
        return tf.clip_by_value(tensor_in, min_value, max_value)

    def tile(self, tensor_in, repeats):
        """
        Repeat tensor data along a specific dimension

        Example:

            >>> import pyhf
            >>> import tensorflow as tf
            >>> sess = tf.Session()
            ...
            >>> pyhf.set_backend(pyhf.tensor.tensorflow_backend(session=sess))
            >>> a = pyhf.tensorlib.astensor([[1.0], [2.0]])
            >>> with sess.as_default():
            ...   sess.run(pyhf.tensorlib.tile(a, (1, 2)))
            ...
            array([[1., 1.],
                   [2., 2.]], dtype=float32)

        Args:
            tensor_in (`Tensor`): The tensor to be repeated
            repeats (`Tensor`): The tuple of multipliers for each dimension

        Returns:
            TensorFlow Tensor: The tensor with repeated axes
        """
        return tf.tile(tensor_in, repeats)

    def conditional(self, predicate, true_callable, false_callable):
        """
        Runs a callable conditional on the boolean value of the evaulation of a predicate

        Example:

            >>> import pyhf
            >>> import tensorflow as tf
            >>> sess = tf.Session()
            ...
            >>> pyhf.set_backend(pyhf.tensor.tensorflow_backend(session=sess))
            >>> tensorlib = pyhf.tensorlib
            >>> a = tensorlib.astensor([4])
            >>> b = tensorlib.astensor([5])
            >>> compare = tensorlib.conditional((a < b)[0], lambda: a + b, lambda: a - b)
            >>> with sess.as_default():
            ...     sess.run(compare)
            ...
            array([9.], dtype=float32)

        Args:
            predicate (`scalar`): The logical condition that determines which callable to evaluate
            true_callable (`callable`): The callable that is evaluated when the :code:`predicate` evalutes to :code:`true`
            false_callable (`callable`): The callable that is evaluated when the :code:`predicate` evalutes to :code:`false`

        Returns:
            TensorFlow Tensor: The output of the callable that was evaluated
        """
        return tf.cond(predicate, true_callable, false_callable)

    def tolist(self, tensor_in):
        try:
            return self.session.run(tensor_in).tolist()
        except AttributeError as err:
            if isinstance(tensor_in, list):
                return tensor_in
            if "no attribute 'run'" in str(err):
                raise RuntimeError(
                    'evaluation of tensor requested via .tolist() but no session defined'
                )
            raise
        except RuntimeError as err:
            # if no tensor operations have been added to the graph, but we want
            # to pass-through a list, then we need to catch the runtime error
            # First, see if the input tensor is just a vanilla python list and
            # return it instead
            if "graph is empty" in str(err) and isinstance(tensor_in, list):
                return tensor_in
            raise
        except TypeError:
            # if a tensor operation has been added to the graph, but we want to
            # pass-through a list, we need to catch the type error
            if isinstance(tensor_in, list):
                return tensor_in
            raise

    def outer(self, tensor_in_1, tensor_in_2):
        tensor_in_1 = (
            tensor_in_1
            if tensor_in_1.dtype != tf.bool
            else tf.cast(tensor_in_1, tf.float32)
        )
        tensor_in_1 = (
            tensor_in_1
            if tensor_in_2.dtype != tf.bool
            else tf.cast(tensor_in_2, tf.float32)
        )
        return tf.einsum('i,j->ij', tensor_in_1, tensor_in_2)

    def gather(self, tensor, indices):
        return tf.compat.v2.gather(tensor, indices)

    def boolean_mask(self, tensor, mask):
        return tf.boolean_mask(tensor, mask)

    def isfinite(self, tensor):
        return tf.math.is_finite(tensor)

    def astensor(self, tensor_in, dtype='float'):
        """
        Convert to a TensorFlow Tensor.

        Args:
            tensor_in (Number or Tensor): Tensor object

        Returns:
            `tf.Tensor`: A symbolic handle to one of the outputs of a `tf.Operation`.
        """
        dtypemap = {'float': tf.float32, 'int': tf.int32, 'bool': tf.bool}
        try:
            dtype = dtypemap[dtype]
        except KeyError:
            log.error('Invalid dtype: dtype must be float, int, or bool.')
            raise

        tensor = tensor_in
        # If already a tensor then done
        try:
            tensor.op
        except AttributeError:
            tensor = tf.convert_to_tensor(tensor_in)
            # Ensure non-empty tensor shape for consistency
            try:
                tensor.shape[0]
            except IndexError:
                tensor = tf.reshape(tensor, [1])
        if tensor.dtype is not dtype:
            tensor = tf.cast(tensor, dtype)
        return tensor

    def sum(self, tensor_in, axis=None):
        return (
            tf.reduce_sum(tensor_in)
            if (axis is None or tensor_in.shape == tf.TensorShape([]))
            else tf.reduce_sum(tensor_in, axis)
        )

    def product(self, tensor_in, axis=None):
        return (
            tf.reduce_prod(tensor_in)
            if axis is None
            else tf.reduce_prod(tensor_in, axis)
        )

    def abs(self, tensor):
        return tf.abs(tensor)

    def ones(self, shape):
        return tf.ones(shape)

    def zeros(self, shape):
        return tf.zeros(shape)

    def power(self, tensor_in_1, tensor_in_2):
        return tf.pow(tensor_in_1, tensor_in_2)

    def sqrt(self, tensor_in):
        return tf.sqrt(tensor_in)

    def shape(self, tensor):
        return tuple(map(int, tensor.shape))

    def reshape(self, tensor, newshape):
        return tf.reshape(tensor, newshape)

    def divide(self, tensor_in_1, tensor_in_2):
        return tf.divide(tensor_in_1, tensor_in_2)

    def log(self, tensor_in):
        return tf.math.log(tensor_in)

    def exp(self, tensor_in):
        return tf.exp(tensor_in)

    def stack(self, sequence, axis=0):
        return tf.stack(sequence, axis=axis)

    def where(self, mask, tensor_in_1, tensor_in_2):
        return tf.compat.v2.where(mask, tensor_in_1, tensor_in_2)

    def concatenate(self, sequence, axis=0):
        """
        Join a sequence of arrays along an existing axis.

        Args:
            sequence: sequence of tensors
            axis: dimension along which to concatenate

        Returns:
            output: the concatenated tensor

        """
        return tf.concat(sequence, axis=axis)

    def simple_broadcast(self, *args):
        """
        Broadcast a sequence of 1 dimensional arrays.

        Example:

            >>> import pyhf
            >>> import tensorflow as tf
            >>> sess = tf.Session()
            ...
            >>> pyhf.set_backend(pyhf.tensor.tensorflow_backend(session=sess))
            >>> tf.Session().run(pyhf.tensorlib.simple_broadcast(
            ...   pyhf.tensorlib.astensor([1]),
            ...   pyhf.tensorlib.astensor([2, 3, 4]),
            ...   pyhf.tensorlib.astensor([5, 6, 7])))
            [array([1., 1., 1.], dtype=float32), array([2., 3., 4.], dtype=float32), array([5., 6., 7.], dtype=float32)]

        Args:
            args (Array of Tensors): Sequence of arrays

        Returns:
            list of Tensors: The sequence broadcast together.
        """
        max_dim = max(map(lambda arg: arg.shape[0], args))
        try:
            assert not [arg for arg in args if 1 < arg.shape[0] < max_dim]
        except AssertionError as error:
            log.error(
                'ERROR: The arguments must be of compatible size: 1 or %i', max_dim
            )
            raise error

        broadcast = [
            arg
            if arg.shape[0] > 1
            else tf.tile(tf.slice(arg, [0], [1]), tf.stack([max_dim]))
            for arg in args
        ]
        return broadcast

    def einsum(self, subscripts, *operands):
        """
        A generalized contraction between tensors of arbitrary dimension.

        This function returns a tensor whose elements are defined by equation,
        which is written in a shorthand form inspired by the Einstein summation
        convention.

        Args:
            subscripts: str, specifies the subscripts for summation
            operands: list of array_like, these are the tensors for the operation

        Returns:
            TensorFlow Tensor: the calculation based on the Einstein summation convention
        """
        return tf.einsum(subscripts, *operands)

    def poisson_logpdf(self, n, lam):
        r"""
        The log of the continous approximation, using :math:`n! = \Gamma\left(n+1\right)`,
        to the probability mass function of the Poisson distribution evaluated
        at :code:`n` given the parameter :code:`lam`.

        Example:

            >>> import pyhf
            >>> import tensorflow as tf
            >>> sess = tf.Session()
            >>> pyhf.set_backend(pyhf.tensor.tensorflow_backend(session=sess))
            ...
            >>> with sess.as_default():
            ...     sess.run(pyhf.tensorlib.poisson_logpdf(5., 6.))
            ...
            -1.8286943
            >>> values = pyhf.tensorlib.astensor([5., 9.])
            >>> rates = pyhf.tensorlib.astensor([6., 8.])
            >>> with sess.as_default():
            ...     sess.run(pyhf.tensorlib.poisson_logpdf(values, rates))
            ...
            array([-1.8286943, -2.086854 ], dtype=float32)

        Args:
            n (`tensor` or `float`): The value at which to evaluate the approximation to the Poisson distribution p.m.f.
                                  (the observed number of events)
            lam (`tensor` or `float`): The mean of the Poisson distribution p.m.f.
                                    (the expected number of events)

        Returns:
            TensorFlow Tensor: Value of the continous approximation to log(Poisson(n|lam))
        """
        return tfp.distributions.Poisson(lam).log_prob(n)

    def poisson(self, n, lam):
        r"""
        The continous approximation, using :math:`n! = \Gamma\left(n+1\right)`,
        to the probability mass function of the Poisson distribution evaluated
        at :code:`n` given the parameter :code:`lam`.

        Example:

            >>> import pyhf
            >>> import tensorflow as tf
            >>> sess = tf.Session()
            >>> pyhf.set_backend(pyhf.tensor.tensorflow_backend(session=sess))
            ...
            >>> with sess.as_default():
            ...     sess.run(pyhf.tensorlib.poisson(5., 6.))
            ...
            0.16062315
            >>> values = pyhf.tensorlib.astensor([5., 9.])
            >>> rates = pyhf.tensorlib.astensor([6., 8.])
            >>> with sess.as_default():
            ...     sess.run(pyhf.tensorlib.poisson(values, rates))
            ...
            array([0.16062315, 0.12407687], dtype=float32)

        Args:
            n (`tensor` or `float`): The value at which to evaluate the approximation to the Poisson distribution p.m.f.
                                  (the observed number of events)
            lam (`tensor` or `float`): The mean of the Poisson distribution p.m.f.
                                    (the expected number of events)

        Returns:
            TensorFlow Tensor: Value of the continous approximation to Poisson(n|lam)
        """
        return tf.exp(tfp.distributions.Poisson(lam).log_prob(n))

    def normal_logpdf(self, x, mu, sigma):
        r"""
        The log of the probability density function of the Normal distribution evaluated
        at :code:`x` given parameters of mean of :code:`mu` and standard deviation
        of :code:`sigma`.

        Example:

            >>> import pyhf
            >>> import tensorflow as tf
            >>> sess = tf.Session()
            >>> pyhf.set_backend(pyhf.tensor.tensorflow_backend(session=sess))
            ...
            >>> with sess.as_default():
            ...     sess.run(pyhf.tensorlib.normal_logpdf(0.5, 0., 1.))
            ...
            -1.0439385
            >>> values = pyhf.tensorlib.astensor([0.5, 2.0])
            >>> means = pyhf.tensorlib.astensor([0., 2.3])
            >>> sigmas = pyhf.tensorlib.astensor([1., 0.8])
            >>> with sess.as_default():
            ...     sess.run(pyhf.tensorlib.normal_logpdf(values, means, sigmas))
            ...
            array([-1.0439385, -0.7661075], dtype=float32)

        Args:
            x (`tensor` or `float`): The value at which to evaluate the Normal distribution p.d.f.
            mu (`tensor` or `float`): The mean of the Normal distribution
            sigma (`tensor` or `float`): The standard deviation of the Normal distribution

        Returns:
            TensorFlow Tensor: Value of log(Normal(x|mu, sigma))
        """
        normal = tfp.distributions.Normal(mu, sigma)
        return normal.log_prob(x)

    def normal(self, x, mu, sigma):
        r"""
        The probability density function of the Normal distribution evaluated
        at :code:`x` given parameters of mean of :code:`mu` and standard deviation
        of :code:`sigma`.

        Example:

            >>> import pyhf
            >>> import tensorflow as tf
            >>> sess = tf.Session()
            >>> pyhf.set_backend(pyhf.tensor.tensorflow_backend(session=sess))
            ...
            >>> with sess.as_default():
            ...     sess.run(pyhf.tensorlib.normal(0.5, 0., 1.))
            ...
            0.35206532
            >>> values = pyhf.tensorlib.astensor([0.5, 2.0])
            >>> means = pyhf.tensorlib.astensor([0., 2.3])
            >>> sigmas = pyhf.tensorlib.astensor([1., 0.8])
            >>> with sess.as_default():
            ...     sess.run(pyhf.tensorlib.normal(values, means, sigmas))
            ...
            array([0.35206532, 0.46481887], dtype=float32)

        Args:
            x (`tensor` or `float`): The value at which to evaluate the Normal distribution p.d.f.
            mu (`tensor` or `float`): The mean of the Normal distribution
            sigma (`tensor` or `float`): The standard deviation of the Normal distribution

        Returns:
            TensorFlow Tensor: Value of Normal(x|mu, sigma)
        """
        normal = tfp.distributions.Normal(mu, sigma)
        return normal.prob(x)

    def normal_cdf(self, x, mu=0, sigma=1):
        """
        The cumulative distribution function for the Normal distribution

        Example:

            >>> import pyhf
            >>> import tensorflow as tf
            >>> sess = tf.Session()
            ...
            >>> pyhf.set_backend(pyhf.tensor.tensorflow_backend(session=sess))
            >>> with sess.as_default():
            ...   sess.run(pyhf.tensorlib.normal_cdf(0.8))
            ...
            0.7881446
            >>> values = pyhf.tensorlib.astensor([0.8, 2.0])
            >>> with sess.as_default():
            ...   sess.run(pyhf.tensorlib.normal_cdf(values))
            ...
            array([0.7881446 , 0.97724986], dtype=float32)

        Args:
            x (`tensor` or `float`): The observed value of the random variable to evaluate the CDF for
            mu (`tensor` or `float`): The mean of the Normal distribution
            sigma (`tensor` or `float`): The standard deviation of the Normal distribution

        Returns:
            TensorFlow Tensor: The CDF
        """
        normal = tfp.distributions.Normal(mu, sigma)
        return normal.cdf(x)

    def poisson_dist(self, rate):
        r"""
        The Poisson distribution with rate parameter :code:`rate`.

        Example:

            >>> import pyhf
            >>> import tensorflow as tf
            >>> sess = tf.Session()
            ...
            >>> pyhf.set_backend(pyhf.tensor.tensorflow_backend(session=sess))
            >>> rates = pyhf.tensorlib.astensor([5, 8])
            >>> values = pyhf.tensorlib.astensor([4, 9])
            >>> poissons = pyhf.tensorlib.poisson_dist(rates)
            >>> with sess.as_default():
            ...   sess.run(poissons.log_prob(values))
            ...
            array([-1.7403021, -2.086854 ], dtype=float32)

        Args:
            rate (`tensor` or `float`): The mean of the Poisson distribution (the expected number of events)

        Returns:
            TensorFlow Probability Poisson distribution: The Poisson distribution class

        """
        return tfp.distributions.Poisson(rate)

    def normal_dist(self, mu, sigma):
        r"""
        The Normal distribution with mean :code:`mu` and standard deviation :code:`sigma`.

        Example:

            >>> import pyhf
            >>> import tensorflow as tf
            >>> sess = tf.Session()
            ...
            >>> pyhf.set_backend(pyhf.tensor.tensorflow_backend(session=sess))
            >>> means = pyhf.tensorlib.astensor([5, 8])
            >>> stds = pyhf.tensorlib.astensor([1, 0.5])
            >>> values = pyhf.tensorlib.astensor([4, 9])
            >>> normals = pyhf.tensorlib.normal_dist(means, stds)
            >>> with sess.as_default():
            ...   sess.run(normals.log_prob(values))
            ...
            array([-1.4189385, -2.2257915], dtype=float32)

        Args:
            mu (`tensor` or `float`): The mean of the Normal distribution
            sigma (`tensor` or `float`): The standard deviation of the Normal distribution

        Returns:
            TensorFlow Probability Normal distribution: The Normal distribution class

        """
        return tfp.distributions.Normal(mu, sigma)
