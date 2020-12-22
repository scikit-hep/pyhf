"""SciPy Optimizer Class."""
from .. import exceptions
from .mixins import OptimizerMixin
import scipy


class scipy_optimizer(OptimizerMixin):
    """
    Optimizer that uses :func:`scipy.optimize.minimize`.
    """

    __slots__ = ['name', 'tolerance']

    def __init__(self, *args, **kwargs):
        """
        Initialize the scipy_optimizer.

        See :class:`pyhf.optimize.mixins.OptimizerMixin` for other configuration options.

        Args:
            tolerance (:obj:`float`): tolerance for termination. See specific optimizer for detailed meaning. Default is None.
        """
        self.name = 'scipy'
        self.tolerance = kwargs.pop('tolerance', None)
        super().__init__(*args, **kwargs)

    def _get_minimizer(
        self, objective_and_grad, init_pars, init_bounds, fixed_vals=None, do_grad=False
    ):
        return scipy.optimize.minimize



    def _custom_internal_minimize(self,objective, init_pars, maxiter = 1000,rtol = 1e-7):
        import jax.experimental.optimizers as optimizers
        import jax
        opt_init, opt_update, opt_getpars = optimizers.adam(step_size = 1e-2)
        state = opt_init(init_pars)
        vold,_ = objective(init_pars)
        def cond(loop_state):
            delta = loop_state['delta']
            i = loop_state['i']
            delta_below  = jax.numpy.logical_and(loop_state['delta'] > 0,loop_state['delta'] < rtol)
            delta_below  = jax.numpy.logical_and(loop_state['i'] > 1, delta_below)
            maxed_iter = loop_state['i'] > maxiter
            return ~jax.numpy.logical_or(maxed_iter,delta_below)
            


        def body(loop_state):
            i = loop_state['i']
            state = loop_state['state'] 
            pars = opt_getpars(state)
            v,g = objective(pars)
            newopt_state = opt_update(0,g,state)

            vold = loop_state['vold']
            delta = jax.numpy.abs(v-vold)/v
            new_state = {}
            new_state['delta'] =  delta
            new_state['state'] =  newopt_state
            new_state['vold'] = v
            new_state['i'] = i+1
            return new_state

        loop_state = {'delta': 0, 'i': 0, 'state': state, 'vold': vold}
        loop_state = jax.lax.while_loop(cond,body,loop_state)

        print('max',loop_state['i'],loop_state['delta'])
        minimized = opt_getpars(loop_state['state'])
        from collections import namedtuple
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
        """
        Same signature as :func:`scipy.optimize.minimize`.

        Minimizer Options:
            maxiter (:obj:`int`): maximum number of iterations. Default is 100000.
            verbose (:obj:`bool`): print verbose output during minimization. Default is off.
            method (:obj:`str`): minimization routine. Default is 'SLSQP'.
            tolerance (:obj:`float`): tolerance for termination. See specific optimizer for detailed meaning. Default is None.

        Returns:
            fitresult (scipy.optimize.OptimizeResult): the fit result
        """
        maxiter = options.pop('maxiter', self.maxiter)
        verbose = options.pop('verbose', self.verbose)
        method = options.pop('method', 'SLSQP')
        tolerance = options.pop('tolerance', self.tolerance)
        if options:
            raise exceptions.Unsupported(
                f"Unsupported options were passed in: {list(options.keys())}."
            )

        fixed_vals = fixed_vals or []
        indices = [i for i, _ in fixed_vals]
        values = [v for _, v in fixed_vals]
        if fixed_vals:
            constraints = [{'type': 'eq', 'fun': lambda v: v[indices] - values}]
            # update the initial values to the fixed value for any fixed parameter
            for idx, fixed_val in fixed_vals:
                x0[idx] = fixed_val
        else:
            constraints = []

        print('bounds', bounds)
        print('constraints', constraints)
        print('jac', do_grad)
        print('init', x0)
        # import torch
        # print('init', func(torch.tensor(x0,requires_grad = True))[0])

        # print('init',func(x0)[0])        

        result = self._custom_internal_minimize(func,x0)
        return result

        result =  minimizer(
            func,
            x0,
            method=method,
            jac=do_grad,
            # bounds=bounds,
            # constraints=constraints,
            # tol=tolerance,
            # options=dict(maxiter=maxiter, disp=bool(verbose)),
        )
        print(result.fun)
        return result
