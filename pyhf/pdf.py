import copy
import logging
log = logging.getLogger(__name__)

from . import get_backend
from . import exceptions
from . import modifiers
from . import utils
from .constraints import gaussian_constraint_combined, poisson_constraint_combined
from .interpolate import _hfinterp_code1,_hfinterp_code0


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
                    self.modifiers.append(modifier_def['name'])
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
        if name not in self.modifiers:
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
            modifier = self.par_map[modifier_def['name']]['modifier']
            if not type(modifier).__name__ == modifier_def['type']:
                raise exceptions.InvalidNameReuse('existing modifier is found, but it is of wrong type {} (instead of {}). Use unique modifier names or use qualify_names=True when constructing the pdf.'.format(type(modifier).__name__, modifier_def['type']))
            return modifier

        # did not return, so create new modifier and return it
        modifier = modifier_cls(sample['data'], modifier_def['data'])
        self.register_paramset(modifier_def['name'], modifier.n_parameters, modifier)

        return modifier

def finalize_stats(modifier):
    tensorlib, _ = get_backend()
    inquad = tensorlib.sqrt(tensorlib.sum(tensorlib.power(tensorlib.astensor(modifier.uncertainties),2), axis=0))
    totals = tensorlib.sum(modifier.nominal_counts,axis=0)
    return tensorlib.divide(inquad,totals)

class Model(object):
    def __init__(self, spec, **config_kwargs):
        self.spec = copy.deepcopy(spec) #may get modified by config
        self.schema = config_kwargs.pop('schema', utils.get_default_schema())
        # run jsonschema validation of input specification against the (provided) schema
        log.info("Validating spec against schema: {0:s}".format(self.schema))
        utils.validate(self.spec, self.schema)
        # build up our representation of the specification
        self.config = _ModelConfig(self.spec, **config_kwargs)

        for m in self.config.modifiers:
            mod = self.config.modifier(m)
            try:
                mod.finalize()
            except AttributeError:
                pass
        self.prepped_constraints_gaussian = gaussian_constraint_combined(self.config)
        self.prepped_constraints_poisson = poisson_constraint_combined(self.config)
        
        _allmods = []
        _allsamples = []
        _allchannels = []
        _allmods = []
        channel_nbins = {}

        for c in self.spec['channels']:
            _allchannels.append(c['name'])
            for s in c['samples']:
                channel_nbins[c['name']] = len(s['data'])
                _allsamples.append(s['name'])
                for mod in s['modifiers']:
                    _allmods.append((mod['name'],mod['type']))
        _allmods = list(set(_allmods))
        _allsamples = list(set(_allsamples))
        _allchannels = list(set(_allchannels))
        self.do_samples  = list(sorted(_allsamples[:]))
        self.do_channels = list(sorted(_allchannels[:]))
        self.do_mods = list(sorted(_allmods[:]))
        self.channel_nbins = channel_nbins
        
        self.finalized_stats = {k:finalize_stats(self.config.modifier(k)) for k,v in self.config.par_map.items() if 'staterror' in k}
        self._make_mega()
        self._prep_mega()

    def _make_mega(self):
        helper = {}
        for c in self.spec['channels']:
            for s in c['samples']:
                helper.setdefault(c['name'],{})[s['name']] = (c,s)

        mega_mods = {}
        import copy
        for m,mtype in self.do_mods:
            for s in self.do_samples:
                modspec = {'type': mtype, 'name': m}
                if mtype == 'histosys':
                    modspec.setdefault('data',{})['hi_data'] = []
                    modspec.setdefault('data',{})['lo_data'] = []
                    modspec.setdefault('data',{})['mask'] = []
                elif mtype == 'normsys':
                    modspec.setdefault('data',{})['hi'] = []
                    modspec.setdefault('data',{})['lo'] = []
                    modspec.setdefault('data',{})['mask'] = []
                elif mtype == 'normfactor':
                    modspec.setdefault('data',{})['mask'] = []
                elif mtype == 'shapefactor':
                    modspec.setdefault('data',{})['mask'] = []
                elif mtype == 'shapesys':
                    modspec.setdefault('data',{})['uncrt'] = []
                    modspec.setdefault('data',{})['mask'] = []
                elif mtype == 'staterror':
                    modspec.setdefault('data',{})['uncrt'] = []
                    modspec.setdefault('data',{})['mask']  = []
                mega_mods.setdefault(s,{})[m] = copy.deepcopy(modspec)
                
        mega_samples = {}
        for s in self.do_samples:
            mega_nom = []
            for c in self.do_channels:
                defined_samp = helper.get(c,{}).get(s)
                defined_samp = None if not defined_samp else defined_samp[1]
                nom = defined_samp['data'] if defined_samp else [0.0]*self.channel_nbins[c]
                mega_nom += nom
                defined_mods = {x['name']:x for x in defined_samp['modifiers']} if defined_samp else {}
                for m,mtype in self.do_mods:
                    thismod = defined_mods.get(m)
                    if mtype == 'histosys':
                        lo_data = thismod['data']['lo_data'] if thismod else nom
                        hi_data = thismod['data']['hi_data'] if thismod else nom
                        maskval = True if thismod else False
                        mega_mods[s][m]['data']['lo_data'] += lo_data
                        mega_mods[s][m]['data']['hi_data'] += hi_data
                        mega_mods[s][m]['data']['mask']    += [maskval]*len(nom) #broadcasting
                        pass
                    elif mtype == 'normsys':
                        maskval = True if thismod else False
                        lo_factor = thismod['data']['lo'] if thismod else 1.0
                        hi_factor = thismod['data']['hi'] if thismod else 1.0
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
        self.mega_samples = mega_samples
        self.mega_mods    = mega_mods

    def _prep_mega(self):
        tensorlib,_ = get_backend()
        self.normsys_histoset = tensorlib.astensor([
            [
                [
                    self.mega_mods[s][m]['data']['lo'],
                    [1.]*len(self.mega_samples[s]['nom']),
                    self.mega_mods[s][m]['data']['hi'],
                ]
                for s in self.do_samples
            ] for m,mtype in self.do_mods if mtype == 'normsys' 
        ])
        self.normsys_mask = tensorlib.astensor([
            [
                [
                    self.mega_mods[s][m]['data']['mask'],
                ]
                for s in self.do_samples
            ] for m,mtype in self.do_mods if mtype == 'normsys' 
        ])
        self.normsys_default = tensorlib.ones(self.normsys_mask.shape)


        self.histosys_histoset = tensorlib.astensor([
            [
                [
                    self.mega_mods[s][m]['data']['lo_data'],
                    self.mega_samples[s]['nom'],
                    self.mega_mods[s][m]['data']['hi_data'],
                ]
                for s in self.do_samples
            ] for m,mtype in self.do_mods if mtype == 'histosys' 
        ])

        self.histosys_mask = tensorlib.astensor([
            [
                [
                    self.mega_mods[s][m]['data']['mask'],
                ]
                for s in self.do_samples
            ] for m,mtype in self.do_mods if mtype == 'histosys' 
        ])
        self.histosys_default = tensorlib.zeros(self.histosys_mask.shape)


        self.normfactor_mask = tensorlib.astensor([
            [
                [
                    self.mega_mods[s][m]['data']['mask'],
                ]
                for s in self.do_samples
            ] for m,mtype in self.do_mods if mtype == 'normfactor' 
        ])
        self.normfactor_default = tensorlib.ones(self.normfactor_mask.shape)

        self.staterror_mask = tensorlib.astensor([
            [
                [
                    self.mega_mods[s][m]['data']['mask'],
                ]
                for s in self.do_samples
            ] for m,mtype in self.do_mods if mtype == 'staterror' 
        ])
        self.staterror_default = tensorlib.ones(self.staterror_mask.shape)


        self.shapesys_mask = tensorlib.astensor([
            [
                [
                    self.mega_mods[s][m]['data']['mask'],
                ]
                for s in self.do_samples
            ] for m,mtype in self.do_mods if mtype == 'shapesys' 
        ])
        self.shapesys_default = tensorlib.ones(self.shapesys_mask.shape)


        parindices = list(range(len(self.config.suggested_init())))
        self.histo_indices = tensorlib.astensor([
            parindices[self.config.par_slice(m)] for m,mtype in self.do_mods if mtype == 'histosys'
        ], dtype='int')

        self.normsys_indices = tensorlib.astensor([
            parindices[self.config.par_slice(m)] for m,mtype in self.do_mods if mtype == 'normsys'
        ], dtype='int')

        self.normfac_indices = tensorlib.astensor([parindices[self.config.par_slice(m)] for m,mtype in self.do_mods if mtype == 'normfactor' ], dtype='int')


        start_index = 0
        channel_slices = []
        for c in self.do_channels:
            end_index = start_index + self.channel_nbins[c]
            channel_slices.append(slice(start_index,end_index))
            start_index = end_index

        binindices = list(range(sum(list(self.channel_nbins.values()))))
        channel_slice_map = {c:binindices[sl] for c,sl in zip(self.do_channels,channel_slices)}

        self.stat_parslices  = [self.config.par_slice(m) for m,mtype in self.do_mods if mtype=='staterror']
        self.stat_targetind  = [channel_slice_map[self.config.modifier(m).channel] for m,mtype in self.do_mods if mtype=='staterror']

        self.shapesys_parslices  = [self.config.par_slice(m) for m,mtype in self.do_mods if mtype=='shapesys']
        self.shapesys_targetind  = [channel_slice_map[self.config.modifier(m).channel] for m,mtype in self.do_mods if mtype=='shapesys']


        thenom = tensorlib.astensor([self.mega_samples[s]['nom'] for s in self.do_samples])
        self.thenom = tensorlib.reshape(thenom,(1,len(self.do_samples),1,sum(list(self.channel_nbins.values()))))


    def expected_auxdata(self, pars):
        # probably more correctly this should be the expectation value of the constraint_pdf
        # or for the constraints we are using (single par constraings with mean == mode), we can
        # just return the alphas

        tensorlib, _ = get_backend()
        # order matters! because we generated auxdata in a certain order
        auxdata = None
        for modname in self.config.auxdata_order:
            thisaux = self.config.param_set(modname).expected_data(
                pars[self.config.par_slice(modname)])
            tocat = [thisaux] if auxdata is None else [auxdata, thisaux]
            auxdata = tensorlib.concatenate(tocat)
        return auxdata

    def _modifications(self,pars):
        tensorlib, _ = get_backend()

        pars = tensorlib.astensor(pars)

        results_norm = None
        if tensorlib.shape(self.normsys_indices)[0]:
            normsys_alphaset = tensorlib.gather(pars,self.normsys_indices)
            results_norm   = _hfinterp_code1(self.normsys_histoset,normsys_alphaset)
            results_norm   = tensorlib.where(self.normsys_mask,results_norm,self.normsys_default)

        results_histo = None
        if tensorlib.shape(self.histo_indices)[0]:
            histosys_alphaset = tensorlib.gather(pars,self.histo_indices)
            results_histo   = _hfinterp_code0(self.histosys_histoset,histosys_alphaset)
            results_histo   = tensorlib.where(self.histosys_mask,results_histo,self.histosys_default)
        
        results_staterr = None
        if len(self.stat_parslices):
            default = [1.]*self.staterror_default.shape[-1]

            factor_row = tensorlib.astensor([
                tensorlib.concatenate((default[:t[0]],pars[sl],default[t[-1]+1:]))
                for sl,t in zip(self.stat_parslices,self.stat_targetind)
            ])

            results_staterr = tensorlib.einsum('s,a,mb->msab',
                    tensorlib.ones(len(self.do_samples)),
                    tensorlib.astensor([1]),
                    factor_row)


            results_staterr = tensorlib.where(self.staterror_mask,results_staterr,self.staterror_default)

        results_shapesys = None
        if len(self.shapesys_parslices):
            #could probably all cols at once 
            #factor columns for each modifier
            default = [1.]*self.shapesys_default.shape[-1]

            # print('what',default)

            factor_row = tensorlib.astensor([
                tensorlib.concatenate((default[:t[0]],pars[sl],default[t[-1]+1:]))
                for sl,t in zip(self.shapesys_parslices,self.shapesys_targetind)
            ])

            results_shapesys = tensorlib.einsum('s,a,mb->msab',
                    tensorlib.ones(len(self.do_samples)),
                    tensorlib.astensor([1]),
                    factor_row)

            results_shapesys = tensorlib.where(self.shapesys_mask,results_shapesys,self.shapesys_default)
            
        results_normfac = None
        if tensorlib.shape(self.normfac_indices)[0]:
            normfactors = tensorlib.gather(pars,self.normfac_indices)
            results_normfac = self.normfactor_mask * tensorlib.reshape(normfactors,tensorlib.shape(normfactors) + (1,1))
            results_normfac = tensorlib.where(self.normfactor_mask,results_normfac,self.normfactor_default)

        deltas  = list(filter(lambda x: x is not None,[results_histo]))
        factors = list(filter(lambda x: x is not None,[
                results_norm,
                results_staterr,
                results_shapesys,
                results_normfac
        ]))
        return deltas, factors

    def expected_actualdata(self,pars):
        deltas, factors = self._modifications(pars)
        
        tensorlib, _ = get_backend()
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
        normal  = self.prepped_constraints_gaussian.logpdf(auxdata,pars)
        poisson = self.prepped_constraints_poisson.logpdf(auxdata,pars)
        return normal + poisson

    def logpdf(self, pars, data):
        tensorlib, _ = get_backend()
        pars, data = tensorlib.astensor(pars), tensorlib.astensor(data)
        cut = tensorlib.shape(data)[0] - len(self.config.auxdata)
        actual_data, aux_data = data[:cut], data[cut:]
        lambdas_data = self.expected_actualdata(pars)
        summands   = tensorlib.log(tensorlib.poisson(actual_data, lambdas_data))
        
        mainpdf    = tensorlib.sum(summands)
        constraint = self.constraint_logpdf(aux_data, pars)
        
        result = mainpdf + constraint
        return tensorlib.astensor(result) * tensorlib.ones((1)) #ensure (1,) array shape also for numpy

    def pdf(self, pars, data):
        tensorlib, _ = get_backend()
        return tensorlib.exp(self.logpdf(pars, data))
