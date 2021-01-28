"""JAX Custom Optimizer Class."""
from .. import exceptions
from .mixins import OptimizerMixin
import scipy


class jaxcustom_optimizer(OptimizerMixin):
    __slots__ = ['name']

    def __init__(self, *args, **kwargs):
        self.name = 'jaxcustom'
        super().__init__(*args, **kwargs)

    def _get_minimizer(
        self, objective_and_grad, init_pars, init_bounds, fixed_vals=None, do_grad=False
    ):
        return None

    def _custom_internal_minimize(self, objective, init_pars, maxiter=1000, rtol=1e-7):
        import jax.experimental.optimizers as optimizers
        import jax

        opt_init, opt_update, opt_getpars = optimizers.adam(step_size=1e-2)
        state = opt_init(init_pars)
        vold, _ = objective(init_pars)

        def cond(loop_state):
            delta = loop_state['delta']
            i = loop_state['i']
            delta_below = jax.numpy.logical_and(
                loop_state['delta'] > 0, loop_state['delta'] < rtol
            )
            delta_below = jax.numpy.logical_and(loop_state['i'] > 1, delta_below)
            maxed_iter = loop_state['i'] > maxiter
            return ~jax.numpy.logical_or(maxed_iter, delta_below)

        def body(loop_state):
            i = loop_state['i']
            state = loop_state['state']
            pars = opt_getpars(state)
            v, g = objective(pars)
            newopt_state = opt_update(0, g, state)
            vold = loop_state['vold']
            delta = jax.numpy.abs(v - vold) / v
            new_state = {}
            new_state['delta'] = delta
            new_state['state'] = newopt_state
            new_state['vold'] = v
            new_state['i'] = i + 1
            return new_state

        loop_state = {'delta': 0, 'i': 0, 'state': state, 'vold': vold}
        # import time
        # start = time.time()
        # # while(cond(loop_state)):
        #     loop_state = body(loop_state)
        loop_state = jax.lax.while_loop(cond, body, loop_state)
        # print(time.time()-start)

        minimized = opt_getpars(loop_state['state'])

        class Result:
            pass

        r = Result()
        r.x = minimized
        r.success = True
        r.fun = objective(minimized)[0]
        return r

    def _minimize(
        self,
        minimizer,
        func,
        x0,
        do_grad=False,
        bounds=None,
        fixed_vals=None,
        return_uncertainties=False,
        options={},
    ):
        assert minimizer == None
        assert fixed_vals == []
        assert return_uncertainties == False
        assert bounds == None
        result = self._custom_internal_minimize(func, x0)
        return result
