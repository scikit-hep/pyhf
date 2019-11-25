import logging
import jsonpatch
from . import exceptions
from . import utils
from .pdf import Model
from .mixins import _ChannelSummaryMixin

logging.basicConfig()
log = logging.getLogger(__name__)


class Workspace(_ChannelSummaryMixin, dict):
    """
    A JSON-serializable object that is built from an object that follows the `workspace.json` schema.
    """

    def __init__(self, spec, **config_kwargs):
        super(Workspace, self).__init__(spec, channels=spec['channels'])
        self.schema = config_kwargs.pop('schema', 'workspace.json')
        self.version = config_kwargs.pop('version', None)
        # run jsonschema validation of input specification against the (provided) schema
        log.info("Validating spec against schema: {0:s}".format(self.schema))
        utils.validate(self, self.schema, version=self.version)

        self.measurement_names = []
        for measurement in self.get('measurements', []):
            self.measurement_names.append(measurement['name'])

        self.observations = {}
        for obs in self['observations']:
            self.observations[obs['name']] = obs['data']

    def __eq__(self, other):
        if not isinstance(other, Workspace):
            return False
        return dict(self) == dict(other)

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return object.__repr__(self)

    # NB: this is a wrapper function to validate the returned measurement object against the spec
    def get_measurement(self, **config_kwargs):
        """
        Get (or create) a measurement object using the following logic:

          1. if the poi name is given, create a measurement object for that poi
          2. if the measurement name is given, find the measurement for the given name
          3. if the measurement index is given, return the measurement at that index
          4. if there are measurements but none of the above have been specified, return the 0th measurement
          5. otherwise, raises `InvalidMeasurement`

        Args:
            poi_name: The name of the parameter of interest to create a new measurement from
            measurement_name: The name of the measurement to use
            measurement_index: The index of the measurement to use

        Returns:
            measurement: A measurement object adhering to the schema defs.json#/definitions/measurement

        """
        m = self._get_measurement(**config_kwargs)
        utils.validate(m, 'measurement.json', self.version)
        return m

    def _get_measurement(self, **config_kwargs):
        """
        See `Workspace::get_measurement`.
        """
        poi_name = config_kwargs.get('poi_name')
        if poi_name:
            return {
                'name': 'NormalMeasurement',
                'config': {'poi': poi_name, 'parameters': []},
            }

        if self.measurement_names:
            measurement_name = config_kwargs.get('measurement_name')
            if measurement_name:
                if measurement_name not in self.measurement_names:
                    log.debug(
                        'measurements defined:\n\t{0:s}'.format(
                            '\n\t'.join(self.measurement_names)
                        )
                    )
                    raise exceptions.InvalidMeasurement(
                        'no measurement by name \'{0:s}\' was found in the workspace, pick from one of the valid ones above'.format(
                            measurement_name
                        )
                    )
                return self['measurements'][
                    self.measurement_names.index(measurement_name)
                ]

            measurement_index = config_kwargs.get('measurement_index')
            if measurement_index:
                return self['measurements'][measurement_index]

            if len(self.measurement_names) > 1:
                log.warning(
                    'multiple measurements defined. Taking the first measurement.'
                )
            return self['measurements'][0]

        raise exceptions.InvalidMeasurement(
            "A measurement was not given to create the Model."
        )

    def model(self, **config_kwargs):
        """
        Create a model object with/without patches applied.

        Args:
            patches: A list of JSON patches to apply to the model specification

        Returns:
            model: A model object adhering to the schema model.json

        """
        measurement = self.get_measurement(**config_kwargs)
        log.debug(
            'model being created for measurement {0:s}'.format(measurement['name'])
        )

        patches = config_kwargs.get('patches', [])

        modelspec = {
            'channels': self['channels'],
            'parameters': measurement['config']['parameters'],
        }
        for patch in patches:
            modelspec = jsonpatch.JsonPatch(patch).apply(modelspec)

        return Model(modelspec, poiname=measurement['config']['poi'], **config_kwargs)

    def data(self, model, with_aux=True):
        """
        Return the data for the supplied model with or without auxiliary data from the model.

        The model is needed as the order of the data depends on the order of the channels in the model.

        Args:
            model: A model object adhering to the schema model.json
            with_aux: Whether to include auxiliary data from the model or not

        Returns:
            data: A list of numbers
        """

        try:
            observed_data = sum(
                (self.observations[c] for c in model.config.channels), []
            )
        except KeyError:
            log.error(
                "Invalid channel: the workspace does not have observation data for one of the channels in the model."
            )
            raise
        if with_aux:
            observed_data += model.config.auxdata
        return observed_data

    def prune(self, modifiers=[], modifier_types=[], samples=[], channels=[]):
        """
        Return a new, pruned workspace specification. This will not modify the original workspace.

        The pruned workspace must also be a valid workspace.

        Args:
          modifiers: A list of modifiers to prune.
          modifier_types: A list of modifier types to prune.
          samples: A list of samples to prune.
          channels: A list of channels to prune.
        """
        newspec = {
            'channels': [
                {
                    'name': channel['name'],
                    'samples': [
                        {
                            'name': sample['name'],
                            'data': sample['data'],
                            'modifiers': [
                                modifier
                                for modifier in sample['modifiers']
                                if modifier['name'] not in modifiers
                                and modifier['type'] not in modifier_types
                            ],
                        }
                        for sample in channel['samples']
                        if sample['name'] not in samples
                    ],
                }
                for channel in self['channels']
                if channel['name'] not in channels
            ],
            'measurements': [
                {
                    'name': measurement['name'],
                    'config': {
                        'parameters': [
                            parameter
                            for parameter in measurement['config']['parameters']
                            if parameter['name'] not in modifiers
                        ],
                        'poi': measurement['config']['poi'],
                    },
                }
                for measurement in self['measurements']
            ],
            'observations': [
                observation
                for observation in self['observations']
                if observation['name'] not in channels
            ],
            'version': self['version'],
        }
        return Workspace(newspec)
