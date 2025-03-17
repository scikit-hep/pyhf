
import sympy.parsing.sympy_parser as parser
import sympy
from pyhf.parameters import ParamViewer
import jax.numpy as jnp
import jax

def create_modifiers(additional_parameters = None):

    class PureFunctionModifierBuilder:
        is_shared = True
        def __init__(self, pdfconfig):
            self.config = pdfconfig
            self.required_parsets = additional_parameters or {}
            self.builder_data = {'local': {},'global': {'symbols': set()}}

        def collect(self, thismod, nom):
            maskval = True if thismod else False
            mask = [maskval] * len(nom)
            return {'mask': mask}

        def append(self, key, channel, sample, thismod, defined_samp):
            self.builder_data['local'].setdefault(key, {}).setdefault(sample, {}).setdefault('data', {'mask': []})

            nom = (
                defined_samp['data']
                if defined_samp
                else [0.0] * self.config.channel_nbins[channel]
            )
            moddata = self.collect(thismod, nom)
            self.builder_data['local'][key][sample]['data']['mask'] += moddata['mask']

            if thismod is not None:
                formula = thismod['data']['formula']
                parsed = parser.parse_expr(formula)
                free_symbols = parsed.free_symbols
                for x in free_symbols:
                    self.builder_data['global'].setdefault('symbols',set()).add(x)
            else:
                parsed = None
            self.builder_data['local'].setdefault(key,{}).setdefault(sample,{}).setdefault('channels',{}).setdefault(channel,{})['parsed'] = parsed

        def finalize(self):
            list_of_symbols = [str(x) for x in self.builder_data['global']['symbols']]
            self.builder_data['global']['symbol_names'] = list_of_symbols
            for modname, modspec in self.builder_data['local'].items():
                for sample, samplespec in modspec.items():
                    for channel, channelspec in samplespec['channels'].items():
                        if channelspec['parsed'] is not None:
                            channelspec['jaxfunc'] = sympy.lambdify(list_of_symbols, channelspec['parsed'], 'jax')
                        else:
                            channelspec['jaxfunc'] = lambda *args: 1.0
            return self.builder_data

    class PureFunctionModifierApplicator:
        op_code = 'multiplication'
        name = 'purefunc'

        def __init__(
            self, modifiers=None, pdfconfig=None, builder_data=None, batch_size=None
        ):
            self.builder_data = builder_data
            self.batch_size = batch_size
            self.pdfconfig = pdfconfig
            self.inputs = [str(x) for x in builder_data['global']['symbols']]

            self.keys = [f'{mtype}/{m}' for m, mtype in modifiers]
            self.modifiers = [m for m, _ in modifiers]

            parfield_shape = (
                (self.batch_size, pdfconfig.npars)
                if self.batch_size
                else (pdfconfig.npars,)
            )

            self.param_viewer = ParamViewer(parfield_shape, pdfconfig.par_map, self.inputs)
            self.create_jax_eval()

        def create_jax_eval(self):
            def eval_func(pars):
                return jnp.array([
                    [
                        jnp.concatenate([
                            self.builder_data['local'][m][s]['channels'][c]['jaxfunc'](*pars)*jnp.ones(self.pdfconfig.channel_nbins[c])
                            for c in self.pdfconfig.channels
                        ])
                        for s in self.pdfconfig.samples
                    ]
                    for m in self.keys

                ])
            self.jaxeval = eval_func
        
        def apply_nonbatched(self,pars):
            return jnp.expand_dims(self.jaxeval(pars),2)

        def apply_batched(self,pars):
            return jax.vmap(self.jaxeval, in_axes=(1,), out_axes=2)(pars)

        def apply(self, pars):
            if not self.param_viewer.index_selection:
                return
            if self.batch_size is None:
                par_selection = self.param_viewer.get(pars)
                results_purefunc = self.apply_nonbatched(par_selection)
            else:
                par_selection = self.param_viewer.get(pars)
                results_purefunc = self.apply_batched(par_selection)
            return results_purefunc
    
    return PureFunctionModifierBuilder, PureFunctionModifierApplicator


from pyhf.modifiers import histfactory_set

def enable(new_params = None):
    modifier_set = {}
    modifier_set.update(**histfactory_set)

    builder, applicator = create_modifiers(new_params)

    modifier_set.update(**{
        applicator.name: (builder, applicator)}
    )
    return modifier_set

def new_unconstrained_scalars(new_params):
    param_spec = {
        p['name']: 
        [{
            'paramset_type': 'unconstrained',
            'n_parameters': 1,
            'is_shared': True,
            'inits': (p['init'],),
            'bounds': ((p['min'], p['max']),),
            'is_scalar': True,
            'fixed': False,
        }]
        for p in new_params
    }
    return param_spec