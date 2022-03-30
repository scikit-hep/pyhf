"""Tensorflow Tensor Library Module."""
import logging
import tensorflow as tf
import tensorflow_probability as tfp

log = logging.getLogger(__name__)


class tensorflow_backend:
    """TensorFlow backend for pyhf"""

    __slots__ = ['name', 'precision', 'dtypemap', 'default_do_grad']

    def __init__(self, **kwargs):
        self.name = 'tensorflow'
        self.precision = kwargs.get('precision', '64b')
        self.dtypemap = {
            'float': tf.float64 if self.precision == '64b' else tf.float32,
            'int': tf.int64 if self.precision == '64b' else tf.int32,
            'bool': tf.bool,
        }
        self.default_do_grad = True

    def _setup(self):
        """
        Run any global setups for the tensorflow lib.
        """

    def clip(self, tensor_in, min_value, max_value):
        """
        Clips (limits) the tensor values to be within a specified min and max.

        Example:
            >>> import pyhf
            >>> pyhf.set_backend("tensorflow")
            >>> a = pyhf.tensorlib.astensor([-2, -1, 0, 1, 2])
            >>> t = pyhf.tensorlib.clip(a, -1, 1)
            >>> print(t)
            tf.Tensor([-1. -1.  0.  1.  1.], shape=(5,), dtype=float64)

        Args:
            tensor_in (:obj:`tensor`): The input tensor object
            min_value (:obj:`scalar` or :obj:`tensor` or :obj:`None`): The minimum value to be cliped to
            max_value (:obj:`scalar` or :obj:`tensor` or :obj:`None`): The maximum value to be cliped to

        Returns:
            TensorFlow Tensor: A clipped `tensor`

        """
        if min_value is None:
            min_value = tf.reduce_min(tensor_in)
        if max_value is None:
            max_value = tf.reduce_max(tensor_in)
        return tf.clip_by_value(tensor_in, min_value, max_value)

    def erf(self, tensor_in):
        """
        The error function of complex argument.

        Example:

            >>> import pyhf
            >>> pyhf.set_backend("tensorflow")
            >>> a = pyhf.tensorlib.astensor([-2., -1., 0., 1., 2.])
            >>> t = pyhf.tensorlib.erf(a)
            >>> print(t)
            tf.Tensor([-0.99532227 -0.84270079  0.          0.84270079  0.99532227], shape=(5,), dtype=float64)

        Args:
            tensor_in (:obj:`tensor`): The input tensor object

        Returns:
            TensorFlow Tensor: The values of the error function at the given points.
        """
        return tf.math.erf(tensor_in)

    def erfinv(self, tensor_in):
        """
        The inverse of the error function of complex argument.

        Example:

            >>> import pyhf
            >>> pyhf.set_backend("tensorflow")
            >>> a = pyhf.tensorlib.astensor([-2., -1., 0., 1., 2.])
            >>> t = pyhf.tensorlib.erfinv(pyhf.tensorlib.erf(a))
            >>> print(t)
            tf.Tensor([-2. -1.  0.  1.  2.], shape=(5,), dtype=float64)

        Args:
            tensor_in (:obj:`tensor`): The input tensor object

        Returns:
            TensorFlow Tensor: The values of the inverse of the error function at the given points.
        """
        return tf.math.erfinv(tensor_in)

    def tile(self, tensor_in, repeats):
        """
        Repeat tensor data along a specific dimension

        Example:
            >>> import pyhf
            >>> pyhf.set_backend("tensorflow")
            >>> a = pyhf.tensorlib.astensor([[1.0], [2.0]])
            >>> t = pyhf.tensorlib.tile(a, (1, 2))
            >>> print(t)
            tf.Tensor(
            [[1. 1.]
             [2. 2.]], shape=(2, 2), dtype=float64)

        Args:
            tensor_in (:obj:`tensor`): The tensor to be repeated
            repeats (:obj:`tensor`): The tuple of multipliers for each dimension

        Returns:
            TensorFlow Tensor: The tensor with repeated axes

        """
        try:
            return tf.tile(tensor_in, repeats)
        except tf.errors.InvalidArgumentError:
            shape = tf.shape(tensor_in).numpy().tolist()
            diff = len(repeats) - len(shape)
            if diff < 0:
                raise
            return tf.tile(tf.reshape(tensor_in, [1] * diff + shape), repeats)

    def conditional(self, predicate, true_callable, false_callable):
        """
        Runs a callable conditional on the boolean value of the evaluation of a predicate

        Example:
            >>> import pyhf
            >>> pyhf.set_backend("tensorflow")
            >>> tensorlib = pyhf.tensorlib
            >>> a = tensorlib.astensor([4])
            >>> b = tensorlib.astensor([5])
            >>> t = tensorlib.conditional((a < b)[0], lambda: a + b, lambda: a - b)
            >>> print(t)
            tf.Tensor([9.], shape=(1,), dtype=float64)

        Args:
            predicate (:obj:`scalar`): The logical condition that determines which callable to evaluate
            true_callable (:obj:`callable`): The callable that is evaluated when the :code:`predicate` evaluates to :code:`true`
            false_callable (:obj:`callable`): The callable that is evaluated when the :code:`predicate` evaluates to :code:`false`

        Returns:
            TensorFlow Tensor: The output of the callable that was evaluated

        """
        return tf.cond(predicate, true_callable, false_callable)

    def tolist(self, tensor_in):
        try:
            return tensor_in.numpy().tolist()
        except AttributeError:
            if isinstance(tensor_in, list):
                return tensor_in
            raise

    def outer(self, tensor_in_1, tensor_in_2):
        dtype = self.dtypemap["float"]
        tensor_in_1 = (
            tensor_in_1 if tensor_in_1.dtype != tf.bool else tf.cast(tensor_in_1, dtype)
        )
        tensor_in_1 = (
            tensor_in_1 if tensor_in_2.dtype != tf.bool else tf.cast(tensor_in_2, dtype)
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

        Example:

            >>> import pyhf
            >>> pyhf.set_backend("tensorflow")
            >>> tensor = pyhf.tensorlib.astensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            >>> tensor
            <tf.Tensor: shape=(2, 3), dtype=float64, numpy=
            array([[1., 2., 3.],
                   [4., 5., 6.]])>
            >>> type(tensor)
            <class 'tensorflow.python.framework.ops.EagerTensor'>

        Args:
            tensor_in (Number or Tensor): Tensor object

        Returns:
            `tf.Tensor`: A symbolic handle to one of the outputs of a `tf.Operation`.

        """
        try:
            dtype = self.dtypemap[dtype]
        except KeyError:
            log.error(
                'Invalid dtype: dtype must be float, int, or bool.', exc_info=True
            )
            raise

        tensor = tensor_in
        # If already a tensor then done
        try:
            # Use a tensor attribute that isn't meaningless when eager execution is enabled
            tensor.device
        except AttributeError:
            tensor = tf.convert_to_tensor(tensor_in)
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

    def ones(self, shape, dtype="float"):
        try:
            dtype = self.dtypemap[dtype]
        except KeyError:
            log.error(
                f"Invalid dtype: dtype must be one of {list(self.dtypemap.keys())}.",
                exc_info=True,
            )
            raise

        return tf.ones(shape, dtype=dtype)

    def zeros(self, shape, dtype="float"):
        try:
            dtype = self.dtypemap[dtype]
        except KeyError:
            log.error(
                f"Invalid dtype: dtype must be one of {list(self.dtypemap.keys())}.",
                exc_info=True,
            )
            raise

        return tf.zeros(shape, dtype=dtype)

    def power(self, tensor_in_1, tensor_in_2):
        return tf.pow(tensor_in_1, tensor_in_2)

    def sqrt(self, tensor_in):
        return tf.sqrt(tensor_in)

    def shape(self, tensor):
        return tuple(map(int, tensor.shape))

    def reshape(self, tensor, newshape):
        return tf.reshape(tensor, newshape)

    def ravel(self, tensor):
        """
        Return a flattened view of the tensor, not a copy.

        Example:

            >>> import pyhf
            >>> pyhf.set_backend("tensorflow")
            >>> tensor = pyhf.tensorlib.astensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            >>> t_ravel = pyhf.tensorlib.ravel(tensor)
            >>> print(t_ravel)
            tf.Tensor([1. 2. 3. 4. 5. 6.], shape=(6,), dtype=float64)

        Args:
            tensor (Tensor): Tensor object

        Returns:
            `tf.Tensor`: A flattened array.
        """
        return self.reshape(tensor, -1)

    def divide(self, tensor_in_1, tensor_in_2):
        return tf.divide(tensor_in_1, tensor_in_2)

    def log(self, tensor_in):
        return tf.math.log(tensor_in)

    def exp(self, tensor_in):
        return tf.exp(tensor_in)

    def percentile(self, tensor_in, q, axis=None, interpolation="linear"):
        r"""
        Compute the :math:`q`-th percentile of the tensor along the specified axis.

        Example:

            >>> import pyhf
            >>> pyhf.set_backend("tensorflow")
            >>> a = pyhf.tensorlib.astensor([[10, 7, 4], [3, 2, 1]])
            >>> t = pyhf.tensorlib.percentile(a, 50)
            >>> print(t)
            tf.Tensor(3.5, shape=(), dtype=float64)
            >>> t = pyhf.tensorlib.percentile(a, 50, axis=1)
            >>> print(t)
            tf.Tensor([7. 2.], shape=(2,), dtype=float64)

        Args:
            tensor_in (`tensor`): The tensor containing the data
            q (:obj:`float` or `tensor`): The :math:`q`-th percentile to compute
            axis (`number` or `tensor`): The dimensions along which to compute
            interpolation (:obj:`str`): The interpolation method to use when the
             desired percentile lies between two data points ``i < j``:

                - ``'linear'``: ``i + (j - i) * fraction``, where ``fraction`` is the
                  fractional part of the index surrounded by ``i`` and ``j``.

                - ``'lower'``: ``i``.

                - ``'higher'``: ``j``.

                - ``'midpoint'``: ``(i + j) / 2``.

                - ``'nearest'``: ``i`` or ``j``, whichever is nearest.

        Returns:
            TensorFlow Tensor: The value of the :math:`q`-th percentile of the tensor along the specified axis.

        """
        return tfp.stats.percentile(
            tensor_in, q, axis=axis, interpolation=interpolation
        )

    def stack(self, sequence, axis=0):
        return tf.stack(sequence, axis=axis)

    def where(self, mask, tensor_in_1, tensor_in_2):
        """
        Apply a boolean selection mask to the elements of the input tensors.

        Example:

            >>> import pyhf
            >>> pyhf.set_backend("tensorflow")
            >>> t = pyhf.tensorlib.where(
            ...     pyhf.tensorlib.astensor([1, 0, 1], dtype='bool'),
            ...     pyhf.tensorlib.astensor([1, 1, 1]),
            ...     pyhf.tensorlib.astensor([2, 2, 2]),
            ... )
            >>> print(t)
            tf.Tensor([1. 2. 1.], shape=(3,), dtype=float64)

        Args:
            mask (bool): Boolean mask (boolean or tensor object of booleans)
            tensor_in_1 (Tensor): Tensor object
            tensor_in_2 (Tensor): Tensor object

        Returns:
            TensorFlow Tensor: The result of the mask being applied to the tensors.

        """
        return tf.where(mask, tensor_in_1, tensor_in_2)

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
            >>> pyhf.set_backend("tensorflow")
            >>> b = pyhf.tensorlib.simple_broadcast(
            ...   pyhf.tensorlib.astensor([1]),
            ...   pyhf.tensorlib.astensor([2, 3, 4]),
            ...   pyhf.tensorlib.astensor([5, 6, 7]))
            >>> print([str(t) for t in b]) # doctest: +NORMALIZE_WHITESPACE
            ['tf.Tensor([1. 1. 1.], shape=(3,), dtype=float64)',
             'tf.Tensor([2. 3. 4.], shape=(3,), dtype=float64)',
             'tf.Tensor([5. 6. 7.], shape=(3,), dtype=float64)']

        Args:
            args (Array of Tensors): Sequence of arrays

        Returns:
            list of Tensors: The sequence broadcast together.

        """

        max_dim = max(map(tf.size, args))
        try:
            assert not [arg for arg in args if 1 < tf.size(arg) < max_dim]
        except AssertionError:
            log.error(
                'ERROR: The arguments must be of compatible size: 1 or %i', max_dim
            )
            raise
        return [tf.broadcast_to(arg, (max_dim,)) for arg in args]

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
        The log of the continuous approximation, using :math:`n! = \Gamma\left(n+1\right)`,
        to the probability mass function of the Poisson distribution evaluated
        at :code:`n` given the parameter :code:`lam`.

        Example:
            >>> import pyhf
            >>> pyhf.set_backend("tensorflow")
            >>> t = pyhf.tensorlib.poisson_logpdf(5., 6.)
            >>> print(t) # doctest:+ELLIPSIS
            tf.Tensor(-1.82869439..., shape=(), dtype=float64)
            >>> values = pyhf.tensorlib.astensor([5., 9.])
            >>> rates = pyhf.tensorlib.astensor([6., 8.])
            >>> t = pyhf.tensorlib.poisson_logpdf(values, rates)
            >>> print(t)
            tf.Tensor([-1.8286944 -2.0868536], shape=(2,), dtype=float64)

        Args:
            n (:obj:`tensor` or :obj:`float`): The value at which to evaluate the approximation to the Poisson distribution p.m.f.
                                  (the observed number of events)
            lam (:obj:`tensor` or :obj:`float`): The mean of the Poisson distribution p.m.f.
                                    (the expected number of events)

        Returns:
            TensorFlow Tensor: Value of the continuous approximation to log(Poisson(n|lam))
        """
        lam = self.astensor(lam)
        return tfp.distributions.Poisson(lam).log_prob(n)

    def poisson(self, n, lam):
        r"""
        The continuous approximation, using :math:`n! = \Gamma\left(n+1\right)`,
        to the probability mass function of the Poisson distribution evaluated
        at :code:`n` given the parameter :code:`lam`.

        .. note::

            Though the p.m.f of the Poisson distribution is not defined for
            :math:`\lambda = 0`, the limit as :math:`\lambda \to 0` is still
            defined, which gives a degenerate p.m.f. of

            .. math::

                \lim_{\lambda \to 0} \,\mathrm{Pois}(n | \lambda) =
                \left\{\begin{array}{ll}
                1, & n = 0,\\
                0, & n > 0
                \end{array}\right.

        Example:
            >>> import pyhf
            >>> pyhf.set_backend("tensorflow")
            >>> t = pyhf.tensorlib.poisson(5., 6.)
            >>> print(t) # doctest:+ELLIPSIS
            tf.Tensor(0.16062314..., shape=(), dtype=float64)
            >>> values = pyhf.tensorlib.astensor([5., 9.])
            >>> rates = pyhf.tensorlib.astensor([6., 8.])
            >>> t = pyhf.tensorlib.poisson(values, rates)
            >>> print(t)
            tf.Tensor([0.16062314 0.12407692], shape=(2,), dtype=float64)

        Args:
            n (:obj:`tensor` or :obj:`float`): The value at which to evaluate the approximation to the Poisson distribution p.m.f.
                                  (the observed number of events)
            lam (:obj:`tensor` or :obj:`float`): The mean of the Poisson distribution p.m.f.
                                    (the expected number of events)

        Returns:
            TensorFlow Tensor: Value of the continuous approximation to Poisson(n|lam)
        """
        lam = self.astensor(lam)
        return tf.exp(tfp.distributions.Poisson(lam).log_prob(n))

    def normal_logpdf(self, x, mu, sigma):
        r"""
        The log of the probability density function of the Normal distribution evaluated
        at :code:`x` given parameters of mean of :code:`mu` and standard deviation
        of :code:`sigma`.

        Example:
            >>> import pyhf
            >>> pyhf.set_backend("tensorflow")
            >>> t = pyhf.tensorlib.normal_logpdf(0.5, 0., 1.)
            >>> print(t) # doctest:+ELLIPSIS
            tf.Tensor(-1.04393853..., shape=(), dtype=float64)
            >>> values = pyhf.tensorlib.astensor([0.5, 2.0])
            >>> means = pyhf.tensorlib.astensor([0., 2.3])
            >>> sigmas = pyhf.tensorlib.astensor([1., 0.8])
            >>> t = pyhf.tensorlib.normal_logpdf(values, means, sigmas)
            >>> print(t)
            tf.Tensor([-1.04393853 -0.76610747], shape=(2,), dtype=float64)

        Args:
            x (:obj:`tensor` or :obj:`float`): The value at which to evaluate the Normal distribution p.d.f.
            mu (:obj:`tensor` or :obj:`float`): The mean of the Normal distribution
            sigma (:obj:`tensor` or :obj:`float`): The standard deviation of the Normal distribution

        Returns:
            TensorFlow Tensor: Value of log(Normal(x|mu, sigma))
        """
        mu = self.astensor(mu)
        sigma = self.astensor(sigma)

        return tfp.distributions.Normal(mu, sigma).log_prob(x)

    def normal(self, x, mu, sigma):
        r"""
        The probability density function of the Normal distribution evaluated
        at :code:`x` given parameters of mean of :code:`mu` and standard deviation
        of :code:`sigma`.

        Example:
            >>> import pyhf
            >>> pyhf.set_backend("tensorflow")
            >>> t = pyhf.tensorlib.normal(0.5, 0., 1.)
            >>> print(t) # doctest:+ELLIPSIS
            tf.Tensor(0.35206532..., shape=(), dtype=float64)
            >>> values = pyhf.tensorlib.astensor([0.5, 2.0])
            >>> means = pyhf.tensorlib.astensor([0., 2.3])
            >>> sigmas = pyhf.tensorlib.astensor([1., 0.8])
            >>> t = pyhf.tensorlib.normal(values, means, sigmas)
            >>> print(t)
            tf.Tensor([0.35206533 0.46481887], shape=(2,), dtype=float64)

        Args:
            x (:obj:`tensor` or :obj:`float`): The value at which to evaluate the Normal distribution p.d.f.
            mu (:obj:`tensor` or :obj:`float`): The mean of the Normal distribution
            sigma (:obj:`tensor` or :obj:`float`): The standard deviation of the Normal distribution

        Returns:
            TensorFlow Tensor: Value of Normal(x|mu, sigma)
        """
        mu = self.astensor(mu)
        sigma = self.astensor(sigma)

        return tfp.distributions.Normal(mu, sigma).prob(x)

    def normal_cdf(self, x, mu=0.0, sigma=1):
        """
        Compute the value of cumulative distribution function for the Normal distribution at x.

        Example:
            >>> import pyhf
            >>> pyhf.set_backend("tensorflow")
            >>> t = pyhf.tensorlib.normal_cdf(0.8)
            >>> print(t) # doctest:+ELLIPSIS
            tf.Tensor(0.78814460..., shape=(), dtype=float64)
            >>> values = pyhf.tensorlib.astensor([0.8, 2.0])
            >>> t = pyhf.tensorlib.normal_cdf(values)
            >>> print(t)
            tf.Tensor([0.7881446  0.97724987], shape=(2,), dtype=float64)

        Args:
            x (:obj:`tensor` or :obj:`float`): The observed value of the random variable to evaluate the CDF for
            mu (:obj:`tensor` or :obj:`float`): The mean of the Normal distribution
            sigma (:obj:`tensor` or :obj:`float`): The standard deviation of the Normal distribution

        Returns:
            TensorFlow Tensor: The CDF
        """
        mu = self.astensor(mu)
        sigma = self.astensor(sigma)

        return tfp.distributions.Normal(mu, sigma).cdf(x)

    def poisson_dist(self, rate):
        r"""
        Construct a Poisson distribution with rate parameter :code:`rate`.

        Example:
            >>> import pyhf
            >>> pyhf.set_backend("tensorflow")
            >>> rates = pyhf.tensorlib.astensor([5, 8])
            >>> values = pyhf.tensorlib.astensor([4, 9])
            >>> poissons = pyhf.tensorlib.poisson_dist(rates)
            >>> t = poissons.log_prob(values)
            >>> print(t)
            tf.Tensor([-1.74030218 -2.0868536 ], shape=(2,), dtype=float64)

        Args:
            rate (:obj:`tensor` or :obj:`float`): The mean of the Poisson distribution (the expected number of events)

        Returns:
            TensorFlow Probability Poisson distribution: The Poisson distribution class

        """
        rate = self.astensor(rate)

        return tfp.distributions.Poisson(rate)

    def normal_dist(self, mu, sigma):
        r"""
        Construct a Normal distribution with mean :code:`mu` and standard deviation :code:`sigma`.

        Example:
            >>> import pyhf
            >>> pyhf.set_backend("tensorflow")
            >>> means = pyhf.tensorlib.astensor([5, 8])
            >>> stds = pyhf.tensorlib.astensor([1, 0.5])
            >>> values = pyhf.tensorlib.astensor([4, 9])
            >>> normals = pyhf.tensorlib.normal_dist(means, stds)
            >>> t = normals.log_prob(values)
            >>> print(t)
            tf.Tensor([-1.41893853 -2.22579135], shape=(2,), dtype=float64)

        Args:
            mu (:obj:`tensor` or :obj:`float`): The mean of the Normal distribution
            sigma (:obj:`tensor` or :obj:`float`): The standard deviation of the Normal distribution

        Returns:
            TensorFlow Probability Normal distribution: The Normal distribution class

        """
        mu = self.astensor(mu)
        sigma = self.astensor(sigma)

        return tfp.distributions.Normal(mu, sigma)

    def to_numpy(self, tensor_in):
        """
        Convert the TensorFlow tensor to a :class:`numpy.ndarray`.

        Example:
            >>> import pyhf
            >>> pyhf.set_backend("tensorflow")
            >>> tensor = pyhf.tensorlib.astensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            >>> print(tensor)
            tf.Tensor(
            [[1. 2. 3.]
             [4. 5. 6.]], shape=(2, 3), dtype=float64)
            >>> numpy_ndarray = pyhf.tensorlib.to_numpy(tensor)
            >>> numpy_ndarray
            array([[1., 2., 3.],
                   [4., 5., 6.]])
            >>> type(numpy_ndarray)
            <class 'numpy.ndarray'>

        Args:
            tensor_in (:obj:`tensor`): The input tensor object.

        Returns:
            :class:`numpy.ndarray`: The tensor converted to a NumPy ``ndarray``.

        """
        return tensor_in.numpy()

    def transpose(self, tensor_in):
        """
        Transpose the tensor.

        Example:
            >>> import pyhf
            >>> pyhf.set_backend("tensorflow")
            >>> tensor = pyhf.tensorlib.astensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            >>> print(tensor)
            tf.Tensor(
            [[1. 2. 3.]
             [4. 5. 6.]], shape=(2, 3), dtype=float64)
            >>> tensor_T = pyhf.tensorlib.transpose(tensor)
            >>> print(tensor_T)
            tf.Tensor(
            [[1. 4.]
             [2. 5.]
             [3. 6.]], shape=(3, 2), dtype=float64)

        Args:
            tensor_in (:obj:`tensor`): The input tensor object.

        Returns:
            TensorFlow Tensor: The transpose of the input tensor.

        """
        return tf.transpose(tensor_in)
