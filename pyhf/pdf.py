import copy
import logging
log = logging.getLogger(__name__)

from . import get_backend
from . import exceptions
from . import modifiers
from . import utils


class _ModelConfig(object):
    @classmethod
    def from_spec(cls,spec,poiname = 'mu', qualify_names = False):
        channels = []
        samples = []
        modifiers = []
        # hacky, need to keep track in which order we added the constraints
        # so that we can generate correctly-ordered data
        instance = cls()
        for channel in spec['channels']:
            channels.append(channel['name'])
            for sample in channel['samples']:
                samples.append(sample['name'])
                for modifier_def in sample['modifiers']:
                    if qualify_names:
                        fullname = '{}/{}'.format(modifier_def['type'],modifier_def['name'])
                        if modifier_def['name'] == poiname:
                            poiname = fullname
                        modifier_def['name'] = fullname
                    modifier = instance.add_or_get_modifier(channel, sample, modifier_def)
                    try:
                        modifier.add_sample(channel, sample, modifier_def)
                        modifiers.append(modifier_def['name'])
                    except:
                        pass

        instance.channels = list(set(channels))
        instance.samples = list(set(samples))
        instance.modifiers = list(set(modifiers))
        instance.set_poi(poiname)
        return instance

    def __init__(self):
        # set up all other bookkeeping variables
        self.poi_index = None
        self.par_map = {}
        self.par_order = []
        self.auxdata = []
        self.auxdata_order = []
        self.next_index = 0

    def suggested_init(self):
        init = []
        for name in self.par_order:
            init = init + self.par_map[name]['modifier'].suggested_init
        return init

    def suggested_bounds(self):
        bounds = []
        for name in self.par_order:
            bounds = bounds + self.par_map[name]['modifier'].suggested_bounds
        return bounds

    def par_slice(self, name):
        return self.par_map[name]['slice']

    def modifier(self, name):
        return self.par_map[name]['modifier']

    def set_poi(self,name):
        if name not in self.modifiers:
            raise exceptions.InvalidModel("The paramter of interest '{0:s}' cannot be fit as it is not declared in the model specification.".format(name))
        s = self.par_slice(name)
        assert s.stop-s.start == 1
        self.poi_index = s.start

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
        npars = modifier.n_parameters

        log.info('adding modifier %s (%s new nuisance parameters)', modifier_def['name'], npars)
        sl = slice(self.next_index, self.next_index + npars)
        self.next_index = self.next_index + npars
        self.par_order.append(modifier_def['name'])
        self.par_map[modifier_def['name']] = {
            'slice': sl,
            'modifier': modifier
        }
        if modifier.is_constrained:
            self.auxdata += self.modifier(modifier_def['name']).auxdata
            self.auxdata_order.append(modifier_def['name'])
        return modifier

def prep_mod_data(config,modindex):
    tensorlib, _ = get_backend()
    prepped = {}
    for parname,mod in config.par_map.items():
        mo,sl,cubeindices = mod['modifier'], mod['slice'],modindex[parname]['indices']
        if mo.__class__.__name__ == 'normsys':
            prepdata = []
            for ind in cubeindices:
                atm = mo.at_minus_one[ind['strings'][0]][ind['strings'][1]]
                atp = mo.at_plus_one[ind['strings'][0]][ind['strings'][1]]
                zer = mo.at_zero
                prepdata.append([atm,zer,atp])
            prepdata = np.asarray(prepdata)
            prepped[parname] = prepdata
        if mo.__class__.__name__ == 'histosys':
            prepdata = []
            for ind in cubeindices:
                atm = mo.at_minus_one[ind['strings'][0]][ind['strings'][1]]
                atp = mo.at_plus_one[ind['strings'][0]][ind['strings'][1]]
                zer = mo.at_zero[ind['strings'][0]][ind['strings'][1]]
                prepdata.append([atm,zer,atp])
            prepdata = np.asarray(prepdata)
            prepped[parname] = prepdata
    return prepped

def make_cube(spec):
    tensorlib, _ = get_backend()
    maxsamples = 0
    maxbins = 0
    nchannels = len(spec['channels'])
    for i,(c) in enumerate(spec['channels']):
        maxsamples = max(maxsamples,len(c['samples']))
        for j,s in enumerate(c['samples']):
            maxbins = max(maxbins,len(s['data']))
            pass
    thecube = tensorlib.ones((nchannels,maxsamples,maxbins))

    sampleindex = {}
    modindex = {}
    for i,(c) in enumerate(spec['channels']):
        for j,s in enumerate(c['samples']):
            thecube[i,j] = s['data']
            thecube[i,j] = s['data']
            for m in s['modifiers']:
                modindex.setdefault(m['name'],{}).setdefault('indices',[]).append({'strings': [c['name'],s['name']], 'indices': (i,j)})
                sampleindex.setdefault(c['name'],{})[s['name']] = (i,j)
    return thecube,modindex,(nchannels,maxsamples,maxbins), sampleindex

import numpy as np
def new_hfinterp1(at_minus_one,at_zero,at_plus_one,alphas):
    #warning, alphas must be orderes
    base_negative = np.divide(at_minus_one, at_zero)
    base_positive = np.divide(at_plus_one, at_zero)
    neg_alphas = alphas[alphas < 0]
    pos_alphas = alphas[alphas >= 0]
    expo_negative = -np.tile(neg_alphas,base_negative.shape+(1,)) #was outer
    expo_positive = np.tile(pos_alphas,base_positive.shape+(1,)) #was outer

    bases_negative = np.tile(base_negative,neg_alphas.shape+(1,)*len(base_negative.shape))
    bases_negative = np.einsum('i...->...i',bases_negative)

    bases_positive = np.tile(base_positive,pos_alphas.shape+(1,)*len(base_positive.shape))
    bases_positive = np.einsum('i...->...i',bases_positive)

    res_neg = np.power(bases_negative,expo_negative)
    res_pos = np.power(bases_positive,expo_positive)

    result = np.concatenate([res_neg,res_pos],axis=-1)
    result.shape
    return np.einsum('...ij->...ji',result)

def new_hfinterp0(at_minus_one,at_zero,at_plus_one,alphas):
    #warning, alphas must be orderes
    iplus_izero  = at_plus_one-at_zero
    izero_iminus = at_zero-at_minus_one

    posfac = alphas[alphas >= 0]
    negfac = alphas[alphas < 0]

    w_pos = posfac * np.ones(iplus_izero.shape + posfac.shape)
    r_pos = iplus_izero.reshape(iplus_izero.shape + (1,)) * w_pos

    w_neg = negfac * np.ones(izero_iminus.shape + negfac.shape)
    r_neg = izero_iminus.reshape(izero_iminus.shape + (1,)) * w_neg

    result = np.concatenate([r_neg,r_pos],axis=-1)
    result = np.einsum('...ij->...ji',result)
    result.shape
    return result

def calculate_constraint(bytype):
    tensorlib, _ = get_backend()
    newsummands = None
    for k,c in bytype.items():
        c = tensorlib.astensor(c)
        #warning, call signature depends on pdf_type (2 for pois, 3 for normal)
        pdfval = getattr(tensorlib,k)(c[:,0],c[:,1],c[:,2])
        constraint_term = tensorlib.log(pdfval)
        newsummands = constraint_term if newsummands is None else tensorlib.concatenate([newsummands,constraint_term])
    return tensorlib.sum(newsummands) if newsummands is not None else 0

def expected_actualdata(config,prepped,op_code_counts,maxdims,thecube,modindex,pars,ravel, stack, fast):
    tensorlib, _ = get_backend()
    nfactors  = op_code_counts.get('multiplication',0)
    nsummands = op_code_counts.get('addition',0)

    sumfields = tensorlib.zeros((1+nsummands,)+maxdims)
    #computation is (fac1*fac2*fac3*...*(delta1+delta2+delta3+...+nominal))
    sumfields[0] = thecube #nominal is the base of the sum

    factorfields = tensorlib.ones((1+nfactors,)+maxdims)

    ifactor,isum = 1,1
    for parname,mod in config.par_map.items():
        mo,sl,cubeindices = mod['modifier'], mod['slice'],modindex[parname]['indices']
        is_summand = mo.op_code == 'addition'
        if not is_summand:
            if mo.__class__.__name__ == 'normsys':
                thispars = tensorlib.astensor(pars[config.par_slice(parname)])
                prepdata = prepped[parname]
                results = new_hfinterp1(prepdata[:,0],prepdata[:,1],prepdata[:,2],alphas=thispars)
                for x,ind in zip(results,cubeindices):
                    factorfields[ifactor][ind['indices']] = x
            else:
                thispars = tensorlib.astensor(pars[config.par_slice(parname)])
                for ind in cubeindices:
                    x = mo.apply(*ind['strings'],pars = thispars)
                    if mo.__class__.__name__ == 'normfactor':
                        ndims = len(thecube[ind['indices']])
                        x = tensorlib.astensor([x]*ndims).reshape(ndims)
                    factorfields[ifactor][ind['indices']] = x
        else:
            if mo.__class__.__name__ == 'histosys':
                prepdata = prepped[parname]
                results = new_hfinterp0(prepdata[:,0],prepdata[:,1],prepdata[:,2],alphas=thispars)
                for x,ind in zip(results,cubeindices):
                    sumfields[isum][ind['indices']] = x
            else:
                for ind in cubeindices:
                    x = mo.apply(*ind['strings'],pars = thispars)
                    sumfields[isum][ind['indices']] = x
        if not is_summand:
            ifactor += 1
        else:
            isum += 1

    factorfields[0] = tensorlib.sum(sumfields, axis = 0)
    expected  = tensorlib.product(factorfields,axis=0) #apply modifiers
    if stack:
        expected = tensorlib.sum(expected,axis=1) #stack samples
    if ravel:
        expected = expected.ravel()
    return expected

def finalize_stats(modifier):
    tensorlib, _ = get_backend()
    inquad = tensorlib.sqrt(tensorlib.sum(tensorlib.power(tensorlib.astensor(modifier.uncertainties),2), axis=0))
    totals = tensorlib.sum(modifier.nominal_counts,axis=0)
    return tensorlib.divide(inquad,totals)

class Model(object):
    def __init__(self, spec, **config_kwargs):
        self.spec = copy.deepcopy(spec) #may get modified by config
        self.config = _ModelConfig.from_spec(self.spec,**config_kwargs)

        self.cube, self.modindex, self.maxdims, self.sampleindex = make_cube(self.spec)
        self.prepped_mod = prep_mod_data(self.config,self.modindex)
        log.warning('not sure when we should be finalizing.. this is the fastest though')
        self.finalized_stats = {k:finalize_stats(self.config.modifier(k)) for k,v in self.config.par_map.items() if 'staterror' in k}

        self.schema = config_kwargs.get('schema', utils.get_default_schema())
        # run jsonschema validation of input specification against the (provided) schema
        log.info("Validating spec against schema: {0:s}".format(self.schema))
        utils.validate(self.spec, self.schema)
        # build up our representation of the specification

        self.op_code_counts = {}
        for v in self.config.par_map.values():
            op_code = v['modifier'].op_code
            self.op_code_counts.setdefault(op_code,0)
            self.op_code_counts[op_code] += 1

    def expected_sample(self, channel, sample, pars):
        #can be optimized by s
        return self.expected_actualdata(pars, ravel = False, stack = False)[self.sampleindex[channel][sample]]

    def expected_auxdata(self, pars):
        # probably more correctly this should be the expectation value of the constraint_pdf
        # or for the constraints we are using (single par constraings with mean == mode), we can
        # just return the alphas

        tensorlib, _ = get_backend()
        # order matters! because we generated auxdata in a certain order
        auxdata = None
        for modname in self.config.auxdata_order:
            thisaux = self.config.modifier(modname).expected_data(
                pars[self.config.par_slice(modname)])
            tocat = [thisaux] if auxdata is None else [auxdata, thisaux]
            auxdata = tensorlib.concatenate(tocat)
        return auxdata

    def expected_actualdata(self, pars, new = False, ravel = True, stack = True, fast = False):
        return expected_actualdata(self.config,self.prepped_mod,self.op_code_counts,self.maxdims,self.cube,self.modindex,pars, ravel = ravel, stack = stack, fast = fast)

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
        tensorlib, _ = get_backend()
        start_index = 0
        bytype = {}
        for cname in self.config.auxdata_order:
            modifier, modslice = self.config.modifier(cname), \
                self.config.par_slice(cname)
            modalphas = modifier.alphas(pars[modslice])
            end_index = start_index + int(modalphas.shape[0])
            thisauxdata = auxdata[start_index:end_index]
            start_index = end_index
            if modifier.pdf_type=='normal':
                if modifier.__class__.__name__ in ['histosys','normsys']:
                    kwargs = {'sigma': tensorlib.astensor([1])}
                elif modifier.__class__.__name__ in ['staterror']:
                    kwargs = {'sigma': self.finalized_stats[cname]}
            else:
                kwargs = {}
            callargs = [thisauxdata,modalphas] + [kwargs['sigma'] if kwargs else []]
            bytype.setdefault(modifier.pdf_type,[]).append(callargs)
        return calculate_constraint(bytype)

    def logpdf(self, pars, data):
        tensorlib, _ = get_backend()
        pars, data = tensorlib.astensor(pars), tensorlib.astensor(data)
        cut = int(data.shape[0]) - len(self.config.auxdata)
        actual_data, aux_data = data[:cut], data[cut:]
        lambdas_data = self.expected_actualdata(pars)
        summands = tensorlib.log(tensorlib.poisson(actual_data, lambdas_data))

        result = tensorlib.sum(summands) + self.constraint_logpdf(aux_data, pars)
        return tensorlib.astensor(result) * tensorlib.ones((1)) #ensure (1,) array shape also for numpy

    def pdf(self, pars, data):
        tensorlib, _ = get_backend()
        return tensorlib.exp(self.logpdf(pars, data))
