from pyhf import exceptions


class _BackendRetriever:
    __slots__ = [
        "_array_subtypes",
        "_array_types",
        "jax_backend",
        "numpy_backend",
    ]

    def __init__(self):
        self._array_types = set()
        self._array_subtypes = set()

    def __getattr__(self, name):
        if name == 'numpy_backend':
            from pyhf.tensor.numpy_backend import numpy_backend

            assert numpy_backend
            # for autocomplete and dir() calls
            self.numpy_backend = numpy_backend
            self._array_types.add(numpy_backend.array_type)
            self._array_subtypes.add(numpy_backend.array_subtype)
            return numpy_backend
        elif name == 'jax_backend':
            try:
                from pyhf.tensor.jax_backend import jax_backend

                assert jax_backend
                # for autocomplete and dir() calls
                self.jax_backend = jax_backend
                self._array_types.add(jax_backend.array_type)
                self._array_subtypes.add(jax_backend.array_subtype)
                return jax_backend
            except ImportError as e:
                raise exceptions.ImportBackendError(
                    "There was a problem importing JAX. The jax backend cannot be used.",
                    e,
                )

    @property
    def array_types(self):
        return tuple(self._array_types)

    @property
    def array_subtypes(self):
        return tuple(self._array_subtypes)


BackendRetriever = _BackendRetriever()
__all__ = ['BackendRetriever']
