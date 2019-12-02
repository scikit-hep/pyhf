import iminuit
import logging
import numpy as np

log = logging.getLogger(__name__)


class minuit_optimizer(object):
    def __init__(self, verbose=False, ncall=10000, errordef=1, steps=1000):
        self.verbose = 0
        self.ncall = ncall
        self.errordef = errordef
        self.steps = steps

    def _make_minuit(
        self, objective, data, pdf, init_pars, init_bounds, fixed_vals=None
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
        fixed_vals = fixed_vals or []
        constraints = {}
        for index, value in fixed_vals:
            constraints = {'fix_p{}'.format(index): True}
            initvals['p{}'.format(index)] = value
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

    def minimize(self, objective, data, pdf, init_pars, par_bounds, fixed_vals=None):
        mm = self._make_minuit(objective, data, pdf, init_pars, par_bounds, fixed_vals)
        result = mm.migrad(ncall=self.ncall)
        assert result
        return np.asarray([x[1] for x in mm.values.items()])

    def unconstrained_bestfit(self, objective, data, pdf, init_pars, par_bounds):
        return self.minimize(objective, data, pdf, init_pars, par_bounds)

    def constrained_bestfit(
        self, objective, constrained_mu, data, pdf, init_pars, par_bounds
    ):
        return self.minimize(
            objective,
            data,
            pdf,
            init_pars,
            par_bounds,
            [(pdf.config.poi_index, constrained_mu)],
        )
