from .. import exceptions


class _BackendRetriever(object):
    def __getattr__(self, name):
        if name == 'numpy_backend':
            from .numpy_backend import numpy_backend

            assert numpy_backend
            # for autocomplete and dir() calls
            self.numpy_backend = numpy_backend
            return numpy_backend
        elif name == 'pytorch_backend':
            try:
                from .pytorch_backend import pytorch_backend

                assert pytorch_backend
                # for autocomplete and dir() calls
                self.pytorch_backend = pytorch_backend
                return pytorch_backend
            except ImportError as e:
                raise exceptions.MissingLibraries(
                    "PyTorch is not installed. This backend cannot be imported.", e
                )
        elif name == 'tensorflow_backend':
            try:
                from .tensorflow_backend import tensorflow_backend

                assert tensorflow_backend
                # for autocomplete and dir() calls
                self.tensorflow_backend = tensorflow_backend
                return tensorflow_backend
            except ImportError as e:
                raise exceptions.MissingLibraries(
                    "TensorFlow is not installed. This backend cannot be imported.", e
                )
        elif name == 'mxnet_backend':
            try:
                from .mxnet_backend import mxnet_backend

                assert mxnet_backend
                # for autocomplete and dir() calls
                self.mxnet_backend = mxnet_backend
                return mxnet_backend
            except ImportError as e:
                raise exceptions.MissingLibraries(
                    "MxNet is not installed. This backend cannot be imported.", e
                )


BackendRetriever = _BackendRetriever()
__all__ = ['BackendRetriever']
