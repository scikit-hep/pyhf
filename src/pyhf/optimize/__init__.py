"""Optimizers for Tensor Functions."""

from .. import exceptions


class _OptimizerRetriever(object):
    def __getattr__(self, name):
        if name == 'scipy_optimizer':
            from .opt_scipy import scipy_optimizer

            assert scipy_optimizer
            # hide away one level of the module name
            # pyhf.optimize.scipy_optimizer.scipy_optimizer->pyhf.optimize.scipy_optimizer
            scipy_optimizer.__module__ = __name__
            # for autocomplete and dir() calls
            self.scipy_optimizer = scipy_optimizer
            return scipy_optimizer
        elif name == 'minuit_optimizer':
            try:
                from .opt_minuit import minuit_optimizer

                assert minuit_optimizer
                # hide away one level of the module name
                # pyhf.optimize.minuit_optimizer.minuit_optimizer->pyhf.optimize.minuit_optimizer
                minuit_optimizer.__module__ = __name__
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


OptimizerRetriever = _OptimizerRetriever()
__all__ = ['OptimizerRetriever']
