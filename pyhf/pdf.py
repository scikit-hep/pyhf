import copy
import logging

from . import get_backend, default_backend
from . import exceptions
from . import modifiers
from . import utils
from .constraints import gaussian_constraint_combined, poisson_constraint_combined
from .paramsets import reduce_paramsets_requirements

log = logging.getLogger(__name__)


class _ModelConfig(object):
    def __init__(self, spec, poiname='mu'):
        self.poi_index = None
        self.par_map = {}
        self.par_order = []
        self.auxdata = []
        self.auxdata_order = []
        self.next_index = 0

        # build up a dictionary of the parameter configurations provided by the user
        _paramsets_user_configs = {}
        for parameter in spec.get('parameters', []):
            if parameter['name'] in _paramsets_user_configs:
                raise exceptions.InvalidModel(
                    'Multiple parameter configurations for {} were found.'.format(
                        parameter['name']
                    )
                )
            _paramsets_user_configs[parameter.pop('name')] = parameter

        self.channels = []
        self.samples = []
        self.parameters = []
        self.modifiers = []
        # keep track of the width of each channel (how many bins)
        self.channel_nbins = {}
        # bookkeep all requirements for paramsets we need to build
        _paramsets_requirements = {}
        # need to keep track in which order we added the constraints
        # so that we can generate correctly-ordered data
        for channel in spec['channels']:
            self.channels.append(channel['name'])
            self.channel_nbins[channel['name']] = len(channel['samples'][0]['data'])
            for sample in channel['samples']:
                self.samples.append(sample['name'])
                for modifier_def in sample['modifiers']:
                    # get the paramset requirements for the given modifier. If
                    # modifier does not exist, we'll have a KeyError
                    self.parameters.append(modifier_def['name'])
                    try:
                        paramset_requirements = modifiers.registry[
                            modifier_def['type']
                        ].required_parset(len(sample['data']))
                    except KeyError:
                        log.exception(
                            'Modifier not implemented yet (processing {0:s}). Available modifiers: {1}'.format(
                                modifier_def['type'], modifiers.registry.keys()
                            )
                        )
                        raise exceptions.InvalidModifier()
                    self.modifiers.append(
                        (
                            modifier_def['name'],  # mod name
                            modifier_def['type'],  # mod type
                        )
                    )

                    # check the shareability (e.g. for shapesys for example)
                    is_shared = paramset_requirements['is_shared']
                    if (
                        not (is_shared)
                        and modifier_def['name'] in _paramsets_requirements
                    ):
                        raise ValueError(
                            "Trying to add unshared-paramset but other paramsets exist with the same name."
                        )
                    if is_shared and not (
                        _paramsets_requirements.get(
                            modifier_def['name'], [{'is_shared': True}]
                        )[0]['is_shared']
                    ):
                        raise ValueError(
                            "Trying to add shared-paramset but other paramset of same name is indicated to be unshared."
                        )
                    _paramsets_requirements.setdefault(modifier_def['name'], []).append(
                        paramset_requirements
                    )

        self.channels = sorted(list(set(self.channels)))
        self.samples = sorted(list(set(self.samples)))
        self.parameters = sorted(list(set(self.parameters)))
        self.modifiers = sorted(list(set(self.modifiers)))
        self.channel_nbins = self.channel_nbins
        self._create_and_register_paramsets(
            _paramsets_requirements, _paramsets_user_configs
        )
        self.set_poi(poiname)

    def suggested_init(self):
        init = []
        for name in self.par_order:
            init = init + self.par_map[name]['paramset'].suggested_init
        return init

    def suggested_bounds(self):
        bounds = []
        for name in self.par_order:
            bounds = bounds + self.par_map[name]['paramset'].suggested_bounds
        return bounds

    def par_slice(self, name):
        return self.par_map[name]['slice']

    def param_set(self, name):
        return self.par_map[name]['paramset']

    def set_poi(self, name):
        if name not in [x for x, _ in self.modifiers]:
            raise exceptions.InvalidModel(
                "The parameter of interest '{0:s}' cannot be fit as it is not declared in the model specification.".format(
                    name
                )
            )
        s = self.par_slice(name)
        assert s.stop - s.start == 1
        self.poi_index = s.start

    def _register_paramset(self, param_name, paramset):
        '''allocates n nuisance parameters and stores paramset > modifier map'''
        log.info(
            'adding modifier %s (%s new nuisance parameters)',
            param_name,
            paramset.n_parameters,
        )

        sl = slice(self.next_index, self.next_index + paramset.n_parameters)
        self.next_index = self.next_index + paramset.n_parameters
        self.par_order.append(param_name)
        self.par_map[param_name] = {'slice': sl, 'paramset': paramset}

    def _create_and_register_paramsets(
        self, paramsets_requirements, paramsets_user_configs
    ):
        for param_name, paramset_requirements in reduce_paramsets_requirements(
            paramsets_requirements, paramsets_user_configs
        ).items():
            paramset_type = paramset_requirements.get('paramset_type')
            paramset = paramset_type(**paramset_requirements)
            self._register_paramset(param_name, paramset)


class Model(object):
    def __init__(self, spec, **config_kwargs):
        self.spec = copy.deepcopy(spec)  # may get modified by config
        self.schema = config_kwargs.pop('schema', utils.get_default_schema())
        # run jsonschema validation of input specification against the (provided) schema
        log.info("Validating spec against schema: {0:s}".format(self.schema))
        utils.validate(self.spec, self.schema)
        # build up our representation of the specification
        self.config = _ModelConfig(self.spec, **config_kwargs)

        self._create_nominal_and_modifiers()

        # this is tricky, must happen before constraint
        # terms try to access auxdata but after
        # combined mods have been created that
        # set the aux data
        for k in sorted(self.config.par_map.keys()):
            parset = self.config.param_set(k)
            if hasattr(parset, 'pdf_type'):  # is constrained
                self.config.auxdata += parset.auxdata
                self.config.auxdata_order.append(k)

        self.constraints_gaussian = gaussian_constraint_combined(self.config)
        self.constraints_poisson = poisson_constraint_combined(self.config)

        self._factor_mods = [
            modtype
            for modtype, mod in modifiers.uncombined.items()
            if mod.op_code == 'multiplication'
        ]
        self._delta_mods = [
            modtype
            for modtype, mod in modifiers.uncombined.items()
            if mod.op_code == 'addition'
        ]

    def _create_nominal_and_modifiers(self):
        default_data_makers = {
            'histosys': lambda: {
                'hi_data': [],
                'lo_data': [],
                'nom_data': [],
                'mask': [],
            },
            'lumi': lambda: {'mask': []},
            'normsys': lambda: {'hi': [], 'lo': [], 'nom_data': [], 'mask': []},
            'normfactor': lambda: {'mask': []},
            'shapefactor': lambda: {'mask': []},
            'shapesys': lambda: {'mask': [], 'uncrt': [], 'nom_data': []},
            'staterror': lambda: {'mask': [], 'uncrt': [], 'nom_data': []},
        }

        # the mega-channel will consist of mega-samples that subscribe to
        # mega-modifiers. i.e. while in normal histfactory, each sample might
        # be affected by some modifiers and some not, here we change it so that
        # samples are affected by all modifiers, but we set up the modifier
        # data such that the application of the modifier does not actually
        # change the bin value for bins that are not originally affected by
        # that modifier
        #
        # We don't actually set up the modifier data here for no-ops, but we do
        # set up the entire structure
        mega_mods = {}
        for m, mtype in self.config.modifiers:
            key = '{}/{}'.format(mtype, m)
            for s in self.config.samples:
                mega_mods.setdefault(s, {})[key] = {
                    'type': mtype,
                    'name': m,
                    'data': default_data_makers[mtype](),
                }

        # helper maps channel-name/sample-name to pairs of channel-sample structs
        helper = {}
        for c in self.spec['channels']:
            for s in c['samples']:
                helper.setdefault(c['name'], {})[s['name']] = (c, s)

        mega_samples = {}
        for s in self.config.samples:
            mega_nom = []
            for c in self.config.channels:
                defined_samp = helper.get(c, {}).get(s)
                defined_samp = None if not defined_samp else defined_samp[1]
                # set nominal to 0 for channel/sample if the pair doesn't exist
                nom = (
                    defined_samp['data']
                    if defined_samp
                    else [0.0] * self.config.channel_nbins[c]
                )
                mega_nom += nom
                defined_mods = (
                    {
                        '{}/{}'.format(x['type'], x['name']): x
                        for x in defined_samp['modifiers']
                    }
                    if defined_samp
                    else {}
                )
                for m, mtype in self.config.modifiers:
                    key = '{}/{}'.format(mtype, m)
                    # this is None if modifier doesn't affect channel/sample.
                    thismod = defined_mods.get(key)
                    # print('key',key,thismod['data'] if thismod else None)
                    if mtype == 'histosys':
                        lo_data = thismod['data']['lo_data'] if thismod else nom
                        hi_data = thismod['data']['hi_data'] if thismod else nom
                        maskval = True if thismod else False
                        mega_mods[s][key]['data']['lo_data'] += lo_data
                        mega_mods[s][key]['data']['hi_data'] += hi_data
                        mega_mods[s][key]['data']['nom_data'] += nom
                        mega_mods[s][key]['data']['mask'] += [maskval] * len(
                            nom
                        )  # broadcasting
                        pass
                    elif mtype == 'normsys':
                        maskval = True if thismod else False
                        lo_factor = thismod['data']['lo'] if thismod else 1.0
                        hi_factor = thismod['data']['hi'] if thismod else 1.0
                        mega_mods[s][key]['data']['nom_data'] += [1.0] * len(nom)
                        mega_mods[s][key]['data']['lo'] += [lo_factor] * len(
                            nom
                        )  # broadcasting
                        mega_mods[s][key]['data']['hi'] += [hi_factor] * len(nom)
                        mega_mods[s][key]['data']['mask'] += [maskval] * len(
                            nom
                        )  # broadcasting
                    elif mtype in ['normfactor', 'shapefactor', 'lumi']:
                        maskval = True if thismod else False
                        mega_mods[s][key]['data']['mask'] += [maskval] * len(
                            nom
                        )  # broadcasting
                    elif mtype in ['shapesys', 'staterror']:
                        uncrt = thismod['data'] if thismod else [0.0] * len(nom)
                        maskval = [True if thismod else False] * len(nom)
                        mega_mods[s][key]['data']['mask'] += maskval
                        mega_mods[s][key]['data']['uncrt'] += uncrt
                        mega_mods[s][key]['data']['nom_data'] += nom
                    else:
                        raise RuntimeError(
                            'not sure how to combine {mtype} into the mega-channel'.format(
                                mtype=mtype
                            )
                        )
            sample_dict = {
                'name': 'mega_{}'.format(s),
                'nom': mega_nom,
                'modifiers': list(mega_mods[s].values()),
            }
            mega_samples[s] = sample_dict

        self.mega_mods = mega_mods

        tensorlib, _ = get_backend()
        thenom = default_backend.astensor(
            [mega_samples[s]['nom'] for s in self.config.samples]
        )
        self.thenom = default_backend.reshape(
            thenom,
            (
                1,
                len(self.config.samples),
                1,
                sum(list(self.config.channel_nbins.values())),
            ),
        )
        self.modifiers_appliers = {
            k: c(
                [x for x in self.config.modifiers if x[1] == k],  # x[1] is mtype
                self.config,
                mega_mods,
            )
            for k, c in modifiers.combined.items()
        }

    def expected_auxdata(self, pars):
        tensorlib, _ = get_backend()
        auxdata = None
        for parname in self.config.auxdata_order:
            # order matters! because we generated auxdata in a certain order
            thisaux = self.config.param_set(parname).expected_data(
                pars[self.config.par_slice(parname)]
            )
            tocat = [thisaux] if auxdata is None else [auxdata, thisaux]
            auxdata = tensorlib.concatenate(tocat)
        return auxdata

    def _modifications(self, pars):
        deltas = list(
            filter(
                lambda x: x is not None,
                [self.modifiers_appliers[k].apply(pars) for k in self._delta_mods],
            )
        )
        factors = list(
            filter(
                lambda x: x is not None,
                [self.modifiers_appliers[k].apply(pars) for k in self._factor_mods],
            )
        )

        return deltas, factors

    def expected_actualdata(self, pars):
        """
        For a single channel single sample, we compute

            Pois(d | fac(pars) * (delta(pars) + nom) ) * Gaus(a | pars[is_gaus], sigmas) * Pois(a * cfac | pars[is_poi] * cfac)

        where:
            - delta(pars) is the result of an apply(pars) of combined modifiers
              with 'addition' op_code
            - factor(pars) is the result of apply(pars) of combined modifiers
              with 'multiplication' op_code
            - pars[is_gaus] are the subset of parameters that are constrained by
              gauss (with sigmas accordingly, some of which are computed by
              modifiers)
            - pars[is_pois] are the poissons and their rates (they come with
              their own additional factors unrelated to factor(pars) which are
              also computed by the finalize() of the modifier)

        So in the end we only make 3 calls to pdfs

            1. The main pdf of data and modified rates
            2. All Gaussian constraint as one call
            3. All Poisson constraints as one call
        """
        tensorlib, _ = get_backend()
        pars = tensorlib.astensor(pars)

        deltas, factors = self._modifications(pars)

        allsum = tensorlib.concatenate(deltas + [tensorlib.astensor(self.thenom)])

        nom_plus_delta = tensorlib.sum(allsum, axis=0)
        nom_plus_delta = tensorlib.reshape(
            nom_plus_delta, (1,) + tensorlib.shape(nom_plus_delta)
        )

        allfac = tensorlib.concatenate(factors + [nom_plus_delta])

        newbysample = tensorlib.product(allfac, axis=0)
        newresults = tensorlib.sum(newbysample, axis=0)
        return newresults[0]  # only one alphas

    def expected_data(self, pars, include_auxdata=True):
        tensorlib, _ = get_backend()
        pars = tensorlib.astensor(pars)
        expected_actual = self.expected_actualdata(pars)

        if not include_auxdata:
            return expected_actual
        expected_constraints = self.expected_auxdata(pars)
        tocat = (
            [expected_actual]
            if expected_constraints is None
            else [expected_actual, expected_constraints]
        )
        return tensorlib.concatenate(tocat)

    def constraint_logpdf(self, auxdata, pars):
        normal = self.constraints_gaussian.logpdf(auxdata, pars)
        poisson = self.constraints_poisson.logpdf(auxdata, pars)
        return normal + poisson

    def mainlogpdf(self, maindata, pars):
        tensorlib, _ = get_backend()
        lambdas_data = self.expected_actualdata(pars)
        summands = tensorlib.poisson_logpdf(maindata, lambdas_data)
        tosum = tensorlib.boolean_mask(summands, tensorlib.isfinite(summands))
        mainpdf = tensorlib.sum(tosum)
        return mainpdf

    def logpdf(self, pars, data):
        try:
            tensorlib, _ = get_backend()
            pars, data = tensorlib.astensor(pars), tensorlib.astensor(data)
            cut = tensorlib.shape(data)[0] - len(self.config.auxdata)
            actual_data, aux_data = data[:cut], data[cut:]

            mainpdf = self.mainlogpdf(actual_data, pars)
            constraint = self.constraint_logpdf(aux_data, pars)

            result = mainpdf + constraint
            return result * tensorlib.ones(
                (1)
            )  # ensure (1,) array shape also for numpy
        except:
            log.error(
                'eval failed for data {} pars: {}'.format(
                    tensorlib.tolist(data), tensorlib.tolist(pars)
                )
            )
            raise

    def pdf(self, pars, data):
        tensorlib, _ = get_backend()
        return tensorlib.exp(self.logpdf(pars, data))
