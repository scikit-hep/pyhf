import iminuit
import logging
import numpy as np

log = logging.getLogger(__name__)


class minuit_optimizer(object):
    def __init__(self, verbose=False, ncall=10000, errordef=1, steps=100):
        self.verbose = 0
        self.ncall = ncall
        self.errordef = errordef
        self.steps = steps

    def _make_minuit(
        self, objective, data, pdf, init_pars, init_bounds, constrained_mu=None
    ):
        def f(pars):
            result = objective(pars, data, pdf)
            logpdf = result[0]
            return logpdf

        parnames = ['p{}'.format(i) for i in range(len(init_pars))]
        kw = {'limit_p{}'.format(i): b for i, b in enumerate(init_bounds)}
        initvals = {'p{}'.format(i): v for i, v in enumerate(init_pars)}
        step_sizes = {
            'error_p{}'.format(i): (b[1] - b[0]) / float(self.steps)
            for i, b in enumerate(init_bounds)
        }
        if constrained_mu is not None:
            constraints = {'fix_p{}'.format(pdf.config.poi_index): True}
            initvals['p{}'.format(pdf.config.poi_index)] = constrained_mu
        else:
            constraints = {}
        kwargs = {}
        for d in [kw, constraints, initvals, step_sizes]:
            kwargs.update(**d)
        mm = iminuit.Minuit(
            f,
            print_level=1 if self.verbose else 0,
            errordef=1,
            use_array_call=True,
            forced_parameters=parnames,
            **kwargs
        )
        return mm

    def unconstrained_bestfit(self, objective, data, pdf, init_pars, par_bounds):
        # The Global Fit
        mm = self._make_minuit(objective, data, pdf, init_pars, par_bounds)
        result = mm.migrad(ncall=self.ncall)
        assert result
        return np.asarray([x[1] for x in mm.values.items()])

    def constrained_bestfit(
        self, objective, constrained_mu, data, pdf, init_pars, par_bounds
    ):
        # The Fit Conditions on a specific POI value
        mm = self._make_minuit(
            objective, data, pdf, init_pars, par_bounds, constrained_mu=constrained_mu
        )
        result = mm.migrad(ncall=self.ncall)
        assert result
        return np.asarray([x[1] for x in mm.values.items()])
