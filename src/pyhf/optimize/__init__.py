"""Optimizers for Tensor Functions."""

from .. import exceptions


class _OptimizerRetriever(object):
    def __getattr__(self, name):
        if name == 'scipy_optimizer':
            from .opt_scipy import ScipyOptimizer as scipy_optimizer

            assert scipy_optimizer
            # for autocomplete and dir() calls
            self.scipy_optimizer = scipy_optimizer
            return scipy_optimizer
        elif name == 'minuit_optimizer':
            try:
                from .opt_minuit import MinuitOptimizer as minuit_optimizer

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
