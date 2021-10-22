from pyhf import exceptions


class _BackendRetriever:
    __slots__ = [
        '_array_types',
        'numpy_backend',
        'jax_backend',
        'pytorch_backend',
        'tensorflow_backend',
    ]

    def __init__(self):
        self._array_types = set()

    def __getattr__(self, name):
        if name == 'numpy_backend':
            from pyhf.tensor.numpy_backend import numpy_backend

            assert numpy_backend
            # for autocomplete and dir() calls
            self.numpy_backend = numpy_backend
            self._array_types.add(numpy_backend.array_type)
            return numpy_backend
        elif name == 'jax_backend':
            try:
                from pyhf.tensor.jax_backend import jax_backend

                assert jax_backend
                # for autocomplete and dir() calls
                self.jax_backend = jax_backend
                self._array_types.add(jax_backend.array_type)
                return jax_backend
            except ImportError as e:
                raise exceptions.ImportBackendError(
                    "There was a problem importing JAX. The jax backend cannot be used.",
                    e,
                )
        elif name == 'pytorch_backend':
            try:
                from pyhf.tensor.pytorch_backend import pytorch_backend

                assert pytorch_backend
                # for autocomplete and dir() calls
                self.pytorch_backend = pytorch_backend
                self._array_types.add(pytorch_backend.array_type)
                return pytorch_backend
            except ImportError as e:
                raise exceptions.ImportBackendError(
                    "There was a problem importing PyTorch. The pytorch backend cannot be used.",
                    e,
                )
        elif name == 'tensorflow_backend':
            try:
                from pyhf.tensor.tensorflow_backend import tensorflow_backend

                assert tensorflow_backend
                # for autocomplete and dir() calls
                self.tensorflow_backend = tensorflow_backend
                self._array_types.add(tensorflow_backend.array_type)
                return tensorflow_backend
            except ImportError as e:
                raise exceptions.ImportBackendError(
                    "There was a problem importing TensorFlow. The tensorflow backend cannot be used.",
                    e,
                )

    @property
    def array_types(self):
        return tuple(self._array_types)


BackendRetriever = _BackendRetriever()
__all__ = ['BackendRetriever']
