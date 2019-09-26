import copy
import logging

from . import get_backend, default_backend
from . import exceptions
from . import modifiers
from . import utils
from . import events
from . import probability as prob
from .constraints import gaussian_constraint_combined, poisson_constraint_combined
from .parameters import reduce_paramsets_requirements, ParamViewer
from .tensor.common import TensorViewer

log = logging.getLogger(__name__)


class _ModelConfig(object):
    def __init__(self, spec, **config_kwargs):
        poiname = config_kwargs.get('poiname', 'mu')

        self.par_map = {}
        self.par_order = []
        self.next_index = 0
        self.poi_name = None
        self.poi_index = None
        self.auxdata = []
        self.auxdata_order = []

        default_modifier_settings = {'normsys': {'interpcode': 'code1'}}

        self.modifier_settings = (
            config_kwargs.get('modifier_settings') or default_modifier_settings
        )

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
                if len(sample['data']) != self.channel_nbins[channel['name']]:
                    raise exceptions.InvalidModel(
                        'The sample {0:s} has {1:d} bins, but the channel it belongs to ({2:s}) has {3:d} bins.'.format(
                            sample['name'],
                            len(sample['data']),
                            channel['name'],
                            self.channel_nbins[channel['name']],
                        )
                    )
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
        self._create_and_register_paramsets(
            _paramsets_requirements, _paramsets_user_configs
        )
        self.set_poi(poiname)
        self.npars = len(self.suggested_init())
        self.nauxdata = len(self.auxdata)

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
        self.poi_name = name
        self.poi_index = s.start

    def _register_paramset(self, param_name, paramset):
        """allocates n nuisance parameters and stores paramset > modifier map"""
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


class _ConstraintModel(object):
    """
    Factory class to create pdfs for the constraint terms
    """

    def __init__(self, config, batch_size):
        self.batch_size = batch_size
        self.config = config

        self.constraints_gaussian = gaussian_constraint_combined(
            config, batch_size=self.batch_size
        )
        self.constraints_poisson = poisson_constraint_combined(
            config, batch_size=self.batch_size
        )

        self.viewer_aux = ParamViewer(
            (self.batch_size or 1, self.config.npars),
            self.config.par_map,
            self.config.auxdata_order,
        )

        assert self.constraints_gaussian.batch_size == self.batch_size
        assert self.constraints_poisson.batch_size == self.batch_size

        indices = []
        if self.constraints_gaussian.has_pdf():
            indices.append(self.constraints_gaussian._normal_data)
        if self.constraints_poisson.has_pdf():
            indices.append(self.constraints_poisson._poisson_data)
        if self.has_pdf():
            self.constraints_tv = TensorViewer(indices, self.batch_size)

    def expected_data(self, pars):
        tensorlib, _ = get_backend()
        auxdata = None
        if not self.viewer_aux.index_selection:
            return None
        slice_data = self.viewer_aux.get(pars)
        for parname, sl in zip(self.config.auxdata_order, self.viewer_aux.slices):
            # order matters! because we generated auxdata in a certain order
            thisaux = self.config.param_set(parname).expected_data(
                tensorlib.einsum('ij->ji', slice_data[sl])
            )
            tocat = [thisaux] if auxdata is None else [auxdata, thisaux]
            auxdata = tensorlib.concatenate(tocat, axis=1)
        if self.batch_size is None:
            return auxdata[0]
        return auxdata

    def _dataprojection(self, data):
        tensorlib, _ = get_backend()
        cut = tensorlib.shape(data)[0] - self.config.nauxdata
        return data[cut:]

    def has_pdf(self):
        '''
        Returns:
            flag (`bool`): Whether the model has a constraint term
        '''
        return self.constraints_gaussian.has_pdf() or self.constraints_poisson.has_pdf()

    def make_pdf(self, pars):
        """
        Args:
            pars (`tensor`): The model parameters

        Returns:
            pdf: A distribution object implementing the constraint pdf of HistFactory.
                 Either a Poissonn, a Gaussian or a joint pdf of both depending on the
                 constraints used in the specification.
        """
        pdfobjs = []

        gaussian_pdf = self.constraints_gaussian.make_pdf(pars)
        if gaussian_pdf:
            pdfobjs.append(gaussian_pdf)

        poisson_pdf = self.constraints_poisson.make_pdf(pars)
        if poisson_pdf:
            pdfobjs.append(poisson_pdf)

        if pdfobjs:
            simpdf = prob.Simultaneous(pdfobjs, self.constraints_tv, self.batch_size)
            return simpdf

    def logpdf(self, auxdata, pars):
        """
        Args:
            auxdata (`tensor`): The auxiliary data (a subset of the full data in a HistFactory model)
            pars (`tensor`): The model parameters

        Returns:
            log pdf value: the log of the pdf value
        """
        simpdf = self.make_pdf(pars)
        return simpdf.log_prob(auxdata)


class _MainModel(object):
    """
    Factory class to create pdfs for the main measurement
    """

    def __init__(self, config, mega_mods, nominal_rates, batch_size):
        self.config = config
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
        self.batch_size = batch_size

        self._nominal_rates = default_backend.tile(
            nominal_rates, (1, 1, self.batch_size or 1, 1)
        )

        self.modifiers_appliers = {
            k: c(
                [x for x in config.modifiers if x[1] == k],  # x[1] is mtype
                config,
                mega_mods,
                batch_size=self.batch_size,
                **config.modifier_settings.get(k, {})
            )
            for k, c in modifiers.combined.items()
        }

        self._precompute()
        events.subscribe('tensorlib_changed')(self._precompute)

    def _precompute(self):
        tensorlib, _ = get_backend()
        self.nominal_rates = tensorlib.astensor(self._nominal_rates)

    def has_pdf(self):
        '''
        Returns:
            flag (`bool`): Whether the model has a Main Model component (yes it does)
        '''
        return True

    def make_pdf(self, pars):
        lambdas_data = self.expected_data(pars)
        return prob.Independent(prob.Poisson(lambdas_data))

    def logpdf(self, maindata, pars):
        """
        Args:
            maindata (`tensor`): The main channnel data (a subset of the full data in a HistFactory model)
            pars (`tensor`): The model parameters

        Returns:
            log pdf value: the log of the pdf value
        """
        return self.make_pdf(pars).log_prob(maindata)

    def _dataprojection(self, data):
        tensorlib, _ = get_backend()
        cut = tensorlib.shape(data)[0] - self.config.nauxdata
        return data[:cut]

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

    def expected_data(self, pars):
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
        deltas, factors = self._modifications(pars)

        allsum = tensorlib.concatenate(deltas + [self.nominal_rates])

        nom_plus_delta = tensorlib.sum(allsum, axis=0)
        nom_plus_delta = tensorlib.reshape(
            nom_plus_delta, (1,) + tensorlib.shape(nom_plus_delta)
        )

        allfac = tensorlib.concatenate(factors + [nom_plus_delta])

        newbysample = tensorlib.product(allfac, axis=0)
        newresults = tensorlib.sum(newbysample, axis=0)
        if self.batch_size is None:
            return newresults[0]
        return newresults


class Model(object):
    def __init__(self, spec, batch_size=None, **config_kwargs):
        self.batch_size = batch_size
        self.spec = copy.deepcopy(spec)  # may get modified by config
        self.schema = config_kwargs.pop('schema', 'model.json')
        self.version = config_kwargs.pop('version', None)
        # run jsonschema validation of input specification against the (provided) schema
        log.info("Validating spec against schema: {0:s}".format(self.schema))
        utils.validate(self.spec, self.schema, version=self.version)
        # build up our representation of the specification
        self.config = _ModelConfig(self.spec, **config_kwargs)

        mega_mods, _nominal_rates = self._create_nominal_and_modifiers(
            self.config, self.spec
        )
        self.main_model = _MainModel(
            self.config,
            mega_mods=mega_mods,
            nominal_rates=_nominal_rates,
            batch_size=self.batch_size,
        )

        # this is tricky, must happen before constraint
        # terms try to access auxdata but after
        # combined mods have been created that
        # set the aux data
        for k in sorted(self.config.par_map.keys()):
            parset = self.config.param_set(k)
            if hasattr(parset, 'pdf_type'):  # is constrained
                self.config.auxdata += parset.auxdata
                self.config.auxdata_order.append(k)

        self.constraint_model = _ConstraintModel(
            config=self.config, batch_size=self.batch_size
        )

        cut = self.nominal_rates.shape[-1]
        total_size = cut + len(self.config.auxdata)
        position = list(range(total_size))
        indices = []
        if self.main_model.has_pdf():
            indices.append(position[:cut])
        if self.constraint_model.has_pdf():
            indices.append(position[cut:])
        self.fullpdf_tv = TensorViewer(indices, self.batch_size)

    def _create_nominal_and_modifiers(self, config, spec):
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
        for m, mtype in config.modifiers:
            key = '{}/{}'.format(mtype, m)
            for s in config.samples:
                mega_mods.setdefault(s, {})[key] = {
                    'type': mtype,
                    'name': m,
                    'data': default_data_makers[mtype](),
                }

        # helper maps channel-name/sample-name to pairs of channel-sample structs
        helper = {}
        for c in spec['channels']:
            for s in c['samples']:
                helper.setdefault(c['name'], {})[s['name']] = (c, s)

        mega_samples = {}
        for s in config.samples:
            mega_nom = []
            for c in config.channels:
                defined_samp = helper.get(c, {}).get(s)
                defined_samp = None if not defined_samp else defined_samp[1]
                # set nominal to 0 for channel/sample if the pair doesn't exist
                nom = (
                    defined_samp['data']
                    if defined_samp
                    else [0.0] * config.channel_nbins[c]
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
                for m, mtype in config.modifiers:
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

        nominal_rates = default_backend.astensor(
            [mega_samples[s]['nom'] for s in config.samples]
        )
        _nominal_rates = default_backend.reshape(
            nominal_rates,
            (
                1,  # modifier dimension.. nominal_rates is the base
                len(self.config.samples),
                1,  # alphaset dimension
                sum(list(config.channel_nbins.values())),
            ),
        )

        return mega_mods, _nominal_rates

    def expected_auxdata(self, pars):
        return self.constraint_model.expected_data(pars)

    def _modifications(self, pars):
        return self.main_model._modifications(pars)

    @property
    def nominal_rates(self):
        return self.main_model.nominal_rates

    def expected_actualdata(self, pars):
        return self.main_model.expected_data(pars)

    def expected_data(self, pars, include_auxdata=True):
        tensorlib, _ = get_backend()
        pars = tensorlib.astensor(pars)
        expected_actual = self.main_model.expected_data(pars)

        if not include_auxdata:
            return expected_actual
        expected_constraints = self.expected_auxdata(pars)
        tocat = (
            [expected_actual]
            if expected_constraints is None
            else [expected_actual, expected_constraints]
        )
        if self.batch_size is None:
            return tensorlib.concatenate(tocat, axis=0)
        return tensorlib.concatenate(tocat, axis=1)

    def constraint_logpdf(self, auxdata, pars):
        return self.constraint_model.logpdf(auxdata, pars)

    def mainlogpdf(self, maindata, pars):
        return self.main_model.logpdf(maindata, pars)

    def make_pdf(self, pars):
        """
        Args:
            pars (`tensor`): The model parameters

        Returns:
            pdf: A distribution object implementing the main measurement pdf of HistFactory
        """
        tensorlib, _ = get_backend()

        pdfobjs = []
        mainpdf = self.main_model.make_pdf(pars)
        if mainpdf:
            pdfobjs.append(mainpdf)
        constraintpdf = self.constraint_model.make_pdf(pars)
        if constraintpdf:
            pdfobjs.append(constraintpdf)

        simpdf = prob.Simultaneous(pdfobjs, self.fullpdf_tv, self.batch_size)
        return simpdf

    def logpdf(self, pars, data):
        try:
            tensorlib, _ = get_backend()
            pars, data = tensorlib.astensor(pars), tensorlib.astensor(data)
            # Verify parameter and data shapes
            if pars.shape[-1] != self.config.npars:
                raise exceptions.InvalidPdfParameters(
                    'eval failed as pars has len {} but {} was expected'.format(
                        pars.shape[-1], self.config.npars
                    )
                )

            if data.shape[-1] != self.nominal_rates.shape[-1] + len(
                self.config.auxdata
            ):
                raise exceptions.InvalidPdfData(
                    'eval failed as data has len {} but {} was expected'.format(
                        data.shape[-1],
                        self.nominal_rates.shape[-1] + len(self.config.auxdata),
                    )
                )

            result = self.make_pdf(pars).log_prob(data)

            if (
                not self.batch_size
            ):  # force to be not scalar, should we changed with #522
                return tensorlib.reshape(result, (1,))
            return result
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
