import copy
import logging
log = logging.getLogger(__name__)

from . import get_backend
from . import exceptions
from . import modifiers
from . import utils
from .constraints import gaussian_constraint_combined, poisson_constraint_combined


from .modifiers.combined_mods import (
    normsys_combinedmod,
    histosys_combinedmod,
    normfac_combinedmod,
    staterror_combined,
    shapesys_combined
)

MOD_REGISTRY = {
    'normsys': normsys_combinedmod,
    'histosys': histosys_combinedmod,
    'normfactor': normfac_combinedmod,
    'staterror': staterror_combined,
    'shapesys': shapesys_combined
}


class _ModelConfig(object):
    def __init__(self, spec, poiname = 'mu', qualify_names = False):
        self.poi_index = None
        self.par_map = {}
        self.par_order = []
        self.auxdata = []
        self.auxdata_order = []
        self.next_index = 0

        self.channels = []
        self.samples = []
        self.parameters = []
        self.modifiers = []
        self.channel_nbins = {}
        # hacky, need to keep track in which order we added the constraints
        # so that we can generate correctly-ordered data
        for channel in spec['channels']:
            self.channels.append(channel['name'])
            self.channel_nbins[channel['name']] = len(channel['samples'][0]['data'])
            for sample in channel['samples']:
                self.samples.append(sample['name'])
                # we need to bookkeep a list of modifiers by type so that we
                # can loop over them on a type-by-type basis
                # types like histosys, normsys, etc...
                sample['modifiers_by_type'] = {}
                for modifier_def in sample['modifiers']:
                    self.parameters.append(modifier_def['name'])
                    if qualify_names:
                        fullname = '{}/{}'.format(modifier_def['type'],modifier_def['name'])
                        if modifier_def['name'] == poiname:
                            poiname = fullname
                        modifier_def['name'] = fullname
                    modifier = self.add_or_get_modifier(channel, sample, modifier_def)
                    modifier.add_sample(channel, sample, modifier_def)
                    self.modifiers.append((modifier_def['name'],modifier_def['type']))
                    sample['modifiers_by_type'].setdefault(modifier_def['type'],[]).append(modifier_def['name'])
        self.channels = list(set(self.channels))
        self.samples = list(set(self.samples))
        self.parameters = list(set(self.parameters))
        self.modifiers = list(set(self.modifiers))
        self.channel_nbins = self.channel_nbins
        self.set_poi(poiname)

    def suggested_init(self):
        init = []
        for name in self.par_order:
            init = init + self.par_map[name]['parset'].suggested_init
        return init

    def suggested_bounds(self):
        bounds = []
        for name in self.par_order:
            bounds = bounds + self.par_map[name]['parset'].suggested_bounds
        return bounds

    def par_slice(self, name):
        return self.par_map[name]['slice']

    def param_set(self, name):
        return self.par_map[name]['parset']

    def modifier(self, name):
        return self.par_map[name]['modifier']

    def set_poi(self,name):
        if name not in [x for x,_ in self.modifiers]:
            raise exceptions.InvalidModel("The paramter of interest '{0:s}' cannot be fit as it is not declared in the model specification.".format(name))
        s = self.par_slice(name)
        assert s.stop-s.start == 1
        self.poi_index = s.start

    def register_paramset(self, name, n_parameters, modifier):
        '''allocates n nuisance parameters and stores paramset > modifier map'''
        log.info('adding modifier %s (%s new nuisance parameters)', name, n_parameters)

        parset = modifier.parset

        sl = slice(self.next_index, self.next_index + n_parameters)
        self.next_index = self.next_index + n_parameters
        self.par_order.append(name)
        self.par_map[name] = {
            'slice': sl,
            'parset': parset,
            'modifier': modifier,
        }
        if modifier.is_constrained:
            self.auxdata += parset.auxdata
            self.auxdata_order.append(name)

    def add_or_get_modifier(self, channel, sample, modifier_def):
        """
        Add a new modifier if it does not exist and return it
        or get the existing modifier and return it

        Args:
            channel: current channel object (e.g. from spec)
            sample: current sample object (e.g. from spec)
            modifier_def: current modifier definitions (e.g. from spec)

        Returns:
            modifier object

        """
        # get modifier class associated with modifier type
        try:
            modifier_cls = modifiers.registry[modifier_def['type']]
        except KeyError:
            log.exception('Modifier type not implemented yet (processing {0:s}). Current modifier types: {1}'.format(modifier_def['type'], modifiers.registry.keys()))
            raise exceptions.InvalidModifier()

        # if modifier is shared, check if it already exists and use it
        if modifier_cls.is_shared and modifier_def['name'] in self.par_map:
            log.info('using existing shared, {0:s}constrained modifier (name={1:s}, type={2:s})'.format('' if modifier_cls.is_constrained else 'un', modifier_def['name'], modifier_cls.__name__))
            modifier = self.modifier(modifier_def['name'])
            if not type(modifier).__name__ == modifier_def['type']:
                raise exceptions.InvalidNameReuse('existing modifier is found, but it is of wrong type {} (instead of {}). Use unique modifier names or use qualify_names=True when constructing the pdf.'.format(type(modifier).__name__, modifier_def['type']))
            return modifier

        # did not return, so create new modifier and return it
        modifier = modifier_cls(sample['data'], modifier_def['data'])
        self.register_paramset(modifier_def['name'], modifier.n_parameters, modifier)

        return modifier

class Model(object):
    def __init__(self, spec, **config_kwargs):
        self.spec = copy.deepcopy(spec) #may get modified by config
        self.schema = config_kwargs.pop('schema', utils.get_default_schema())
        # run jsonschema validation of input specification against the (provided) schema
        log.info("Validating spec against schema: {0:s}".format(self.schema))
        utils.validate(self.spec, self.schema)
        # build up our representation of the specification
        self.config = _ModelConfig(self.spec, **config_kwargs)

        self._create_nominal_and_modifiers()

        for m,_ in self.config.modifiers:
            mod = self.config.modifier(m)
            try:
                mod.finalize()
            except AttributeError:
                pass
        self.constraints_gaussian = gaussian_constraint_combined(self.config)
        self.constraints_poisson = poisson_constraint_combined(self.config)


    def _create_nominal_and_modifiers(self):
        default_data_makers = {
            'histosys': lambda: {'hi_data': [], 'lo_data': [], 'nom_data': [],'mask': []},
            'normsys': lambda: {'hi': [], 'lo': [], 'nom_data': [], 'mask': []},
            'shapefactor': lambda: {'mask': []},
            'normfactor': lambda: {'mask': []},
            'shapesys': lambda: {'mask': [], 'uncrt': []},
            'staterror': lambda: {'mask': [], 'uncrt': []},
        }

        mega_mods = {}
        for m,mtype in self.config.modifiers:
            for s in self.config.samples:
                mega_mods.setdefault(s,{})[m] = {
                    'type': mtype,
                    'name': m,
                    'data': default_data_makers[mtype]()
                }

        helper = {}
        for c in self.spec['channels']:
            for s in c['samples']:
                helper.setdefault(c['name'],{})[s['name']] = (c,s)


        mega_samples = {}
        for s in self.config.samples:
            mega_nom = []
            for c in self.config.channels:
                defined_samp = helper.get(c,{}).get(s)
                defined_samp = None if not defined_samp else defined_samp[1]
                nom = defined_samp['data'] if defined_samp else [0.0]*self.config.channel_nbins[c]
                mega_nom += nom
                defined_mods = {x['name']:x for x in defined_samp['modifiers']} if defined_samp else {}
                for m,mtype in self.config.modifiers:
                    thismod = defined_mods.get(m)
                    if mtype == 'histosys':
                        lo_data = thismod['data']['lo_data'] if thismod else nom
                        hi_data = thismod['data']['hi_data'] if thismod else nom
                        maskval = True if thismod else False
                        mega_mods[s][m]['data']['lo_data'] += lo_data
                        mega_mods[s][m]['data']['hi_data'] += hi_data
                        mega_mods[s][m]['data']['nom_data'] += nom
                        mega_mods[s][m]['data']['mask']    += [maskval]*len(nom) #broadcasting
                        pass
                    elif mtype == 'normsys':
                        maskval = True if thismod else False
                        lo_factor = thismod['data']['lo'] if thismod else 1.0
                        hi_factor = thismod['data']['hi'] if thismod else 1.0
                        mega_mods[s][m]['data']['nom_data'] += [1.0]*len(nom) 
                        mega_mods[s][m]['data']['lo']   += [lo_factor]*len(nom) #broadcasting
                        mega_mods[s][m]['data']['hi']   += [hi_factor]*len(nom)
                        mega_mods[s][m]['data']['mask'] += [maskval]  *len(nom) #broadcasting
                    elif mtype == 'normfactor':
                        maskval = True if thismod else False
                        mega_mods[s][m]['data']['mask'] += [maskval]*len(nom) #broadcasting
                    elif mtype == 'shapesys':
                        uncrt = thismod['data'] if thismod else [0.0]*len(nom)
                        maskval = [True if thismod else False]*len(nom)
                        mega_mods[s][m]['data']['mask']  += maskval
                        mega_mods[s][m]['data']['uncrt'] += uncrt
                    elif mtype == 'staterror':
                        uncrt = thismod['data'] if thismod else [0.0]*len(nom)
                        maskval = [True if thismod else False]*len(nom)
                        mega_mods[s][m]['data']['mask']  += maskval
                        mega_mods[s][m]['data']['uncrt'] += uncrt
                    elif mtype == 'shapefactor':
                        maskval = True if thismod else False
                        mega_mods[s][m]['data']['mask'] += [maskval]*len(nom) #broadcasting
                    else:
                        raise RuntimeError('not sure how to combine {mtype} into the mega-channel'.format(mtype = mtype))
            sample_dict = {
                'name': 'mega_{}'.format(s),
                'nom': mega_nom,
                'modifiers': list(mega_mods[s].values())
            }
            mega_samples[s] = sample_dict

        tensorlib,_ = get_backend()
        thenom = tensorlib.astensor(
            [mega_samples[s]['nom'] for s in self.config.samples]
        )
        self.thenom = tensorlib.reshape(thenom,(
            1,
            len(self.config.samples),
            1,
            sum(list(self.config.channel_nbins.values()))
            )
        )
        self.modifiers_appliers = {
            k:c(
                [m for m,mtype in self.config.modifiers if mtype == k],
                self.config,
                mega_mods
            )
            for k,c in MOD_REGISTRY.items()
        }

    def expected_auxdata(self, pars):
        tensorlib, _ = get_backend()
        auxdata = None
        for parname in self.config.auxdata_order:
            # order matters! because we generated auxdata in a certain order
            thisaux = self.config.param_set(parname).expected_data(
                pars[self.config.par_slice(parname)])
            tocat = [thisaux] if auxdata is None else [auxdata, thisaux]
            auxdata = tensorlib.concatenate(tocat)
        return auxdata

    def _modifications(self,pars):
        #shapefactor must be added
        factor_mods = ['normsys','staterror','shapesys','normfactor']
        delta_mods  = ['histosys']

        deltas  = list(filter(lambda x: x is not None,[
            self.modifiers_appliers[k].apply(pars)
            for k in delta_mods
        ]))
        factors = list(filter(lambda x: x is not None,[
            self.modifiers_appliers[k].apply(pars)
            for k in factor_mods
        ]))
        return deltas, factors

    def expected_actualdata(self,pars):
        tensorlib, _ = get_backend()
        pars = tensorlib.astensor(pars)

        deltas, factors = self._modifications(pars)

        allsum = tensorlib.concatenate(deltas + [self.thenom])

        nom_plus_delta = tensorlib.sum(allsum,axis=0)
        nom_plus_delta = tensorlib.reshape(nom_plus_delta,(1,)+tensorlib.shape(nom_plus_delta))

        allfac = tensorlib.concatenate(factors + [nom_plus_delta])

        newbysample = tensorlib.product(allfac,axis=0)
        newresults = tensorlib.sum(newbysample,axis=0)
        return newresults[0] #only one alphas

    def expected_data(self, pars, include_auxdata=True):
        tensorlib, _ = get_backend()
        pars = tensorlib.astensor(pars)
        expected_actual = self.expected_actualdata(pars)

        if not include_auxdata:
            return expected_actual
        expected_constraints = self.expected_auxdata(pars)
        tocat = [expected_actual] if expected_constraints is None else [expected_actual,expected_constraints]
        return tensorlib.concatenate(tocat)

    def constraint_logpdf(self, auxdata, pars):
        normal  = self.constraints_gaussian.logpdf(auxdata,pars)
        poisson = self.constraints_poisson.logpdf(auxdata,pars)
        return normal + poisson

    def mainlogpdf(self, maindata, pars):
        tensorlib, _ = get_backend()
        lambdas_data = self.expected_actualdata(pars)
        summands   = tensorlib.poisson_logpdf(maindata, lambdas_data)
        tosum      = tensorlib.boolean_mask(summands,tensorlib.isfinite(summands))
        mainpdf    = tensorlib.sum(tosum)
        return mainpdf

    def logpdf(self, pars, data):
        try:
            tensorlib, _ = get_backend()
            pars, data = tensorlib.astensor(pars), tensorlib.astensor(data)
            cut = tensorlib.shape(data)[0] - len(self.config.auxdata)
            actual_data, aux_data = data[:cut], data[cut:]

            mainpdf    = self.mainlogpdf(actual_data,pars)
            constraint = self.constraint_logpdf(aux_data, pars)

            result = mainpdf + constraint
            return result * tensorlib.ones((1)) #ensure (1,) array shape also for numpy
        except:
            log.error('eval failed for data {} pars: {}'.format(
                tensorlib.tolist(data),
                tensorlib.tolist(pars)
            ))
            raise

    def pdf(self, pars, data):
        tensorlib, _ = get_backend()
        return tensorlib.exp(self.logpdf(pars, data))
