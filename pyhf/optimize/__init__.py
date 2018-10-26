from .. import exceptions


class _OptimizerRetriever(object):
    def __getattr__(self, name):
        if name == 'scipy_optimizer':
            from .opt_scipy import scipy_optimizer

            assert scipy_optimizer
            self.scipy_optimizer = scipy_optimizer
            return scipy_optimizer
        elif name == 'pytorch_optimizer':
            try:
                from .opt_pytorch import pytorch_optimizer

                assert pytorch_optimizer
                self.pytorch_optimizer = pytorch_optimizer
                return pytorch_optimizer
            except ImportError:
                raise exceptions.MissingLibraries(
                    "PyTorch is not installed. This optimizer cannot be imported."
                )
        elif name == 'tflow_optimizer':
            try:
                from .opt_tflow import tflow_optimizer

                assert tflow_optimizer
                self.tflow_optimizer = tflow_optimizer
                return tflow_optimizer
            except ImportError:
                raise exceptions.MissingLibraries(
                    "TensorFlow is not installed. This optimizer cannot be imported."
                )
        elif name == 'minuit_optimizer':
            try:
                from .opt_minuit import minuit_optimizer

                assert minuit_optimizer
                self.minuit_optimizer = minuit_optimizer
                return minuit_optimizer
            except ImportError:
                raise exceptions.MissingLibraries(
                    "Minuit is not installed. This optimizer cannot be imported."
                )
        else:
            raise exceptions.InvalidOptimizer(
                "Requested optimizer {} does not exist.".format(name)
            )


OptimizerRetriever = _OptimizerRetriever()
__all__ = ['OptimizerRetriever']
