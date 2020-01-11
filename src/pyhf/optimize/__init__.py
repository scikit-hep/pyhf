"""Optimizers for Tensor Functions."""

from .. import exceptions


class _OptimizerRetriever(object):
    def __getattr__(self, name):
        if name == 'scipy_optimizer':
            from .opt_scipy import scipy_optimizer

            assert scipy_optimizer
            # for autocomplete and dir() calls
            self.scipy_optimizer = scipy_optimizer
            return scipy_optimizer
        elif name == 'jax_optimizer':
            try:
                from .opt_jax import jax_optimizer

                assert jax_optimizer
                self.jax_optimizer = jax_optimizer
                return jax_optimizer
            except ImportError as e:
                raise exceptions.ImportBackendError(
                    "There was a problem importing jax. The jax optimizer cannot be used.",
                    e,
                )
        elif name == 'pytorch_optimizer':
            try:
                from .opt_pytorch import pytorch_optimizer

                assert pytorch_optimizer
                # for autocomplete and dir() calls
                self.pytorch_optimizer = pytorch_optimizer
                return pytorch_optimizer
            except ImportError as e:
                raise exceptions.ImportBackendError(
                    "There was a problem importing PyTorch. The pytorch optimizer cannot be used.",
                    e,
                )
        elif name == 'tflow_optimizer':
            try:
                from .opt_tflow import tflow_optimizer

                assert tflow_optimizer
                # for autocomplete and dir() calls
                self.tflow_optimizer = tflow_optimizer
                return tflow_optimizer
            except ImportError as e:
                raise exceptions.ImportBackendError(
                    "There was a problem importing TensorFlow. The tensorflow optimizer cannot be used.",
                    e,
                )
        elif name == 'minuit_optimizer':
            try:
                from .opt_minuit import minuit_optimizer

                assert minuit_optimizer
                # for autocomplete and dir() calls
                self.minuit_optimizer = minuit_optimizer
                return minuit_optimizer
            except ImportError as e:
                raise exceptions.ImportBackendError(
                    "There was a problem importing Minuit. The minuit optimizer cannot be used.",
                    e,
                )
        elif name == '__wrapped__':  # doctest
            pass
        else:
            raise exceptions.InvalidOptimizer(
                "The requested optimizer \"{0:s}\" does not exist.".format(name)
            )


OptimizerRetriever = _OptimizerRetriever()
__all__ = ['OptimizerRetriever']
