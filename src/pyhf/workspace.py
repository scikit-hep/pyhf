"""
pyhf workspaces hold the three data items:

* the statistical model p(data|parameters)
* the observed data (optional)
* fit configurations ("measurements")
"""
import logging
import jsonpatch
import copy
import collections
from . import exceptions
from . import utils
from .pdf import Model
from .mixins import _ChannelSummaryMixin

logging.basicConfig()
log = logging.getLogger(__name__)


def _join_items(join, left_items, right_items, key='name'):
    """
    Join two lists of dictionaries along the given key.

    This is meant to be as generic as possible for any pairs of lists of dictionaries for many join operations.

    Args:
        join (str): The join operation to apply. See ~pyhf.workspace.Workspace for valid join operations.
        left_items (list): A list of dictionaries to join on the left
        right_items (list): A list of dictionaries to join on the right

    Returns:
        :obj:`list`: A joined list of dictionaries.

    """
    if join == 'right outer':
        primary_items, secondary_items = right_items, left_items
    else:
        primary_items, secondary_items = left_items, right_items
    joined_items = copy.deepcopy(primary_items)
    for secondary_item in secondary_items:
        # outer join: merge primary and secondary, matching where possible
        if join == 'outer' and secondary_item in primary_items:
            continue
        # left/right outer join: only add secondary if existing item (by key value) is not in primary
        # NB: this will be slow for large numbers of items
        elif join in ['left outer', 'right outer'] and secondary_item[key] in [
            item[key] for item in joined_items
        ]:
            continue
        joined_items.append(copy.deepcopy(secondary_item))
    return joined_items


def _join_versions(join, left_version, right_version):
    """
    Join two workspace versions.

    Raises:
      ~pyhf.exceptions.InvalidWorkspaceOperation: Versions are incompatible.

    Args:
        join (str): The join operation to apply. See ~pyhf.workspace.Workspace for valid join operations.
        left_version (str): The left workspace version.
        right_version (str): The right workspace version.

    Returns:
        :obj:`str`: The workspace version.

    """
    if left_version != right_version:
        raise exceptions.InvalidWorkspaceOperation(
            f"Workspaces of different versions cannot be combined: {left_version} != {right_version}"
        )
    return left_version


def _join_channels(join, left_channels, right_channels):
    """
    Join two workspace channel specifications.

    Raises:
      ~pyhf.exceptions.InvalidWorkspaceOperation: Channel specifications are incompatible.

    Args:
        join (str): The join operation to apply. See ~pyhf.workspace.Workspace for valid join operations.
        left_channels (list): The left channel specification.
        right_channels (list): The right channel specification.

    Returns:
        :obj:`list`: A joined list of channels. Each channel follows the :obj:`defs.json#channel` `schema <https://scikit-hep.org/pyhf/likelihood.html#channel>`__

    """

    joined_channels = _join_items(join, left_channels, right_channels)
    if join == 'none':
        common_channels = set(c['name'] for c in left_channels).intersection(
            c['name'] for c in right_channels
        )
        if common_channels:
            raise exceptions.InvalidWorkspaceOperation(
                f"Workspaces cannot have any channels in common with the same name: {common_channels}. You can also try a different join operation: {Workspace.valid_joins}."
            )

    elif join == 'outer':
        counted_channels = collections.Counter(
            channel['name'] for channel in joined_channels
        )
        incompatible_channels = [
            channel for channel, count in counted_channels.items() if count > 1
        ]
        if incompatible_channels:
            raise exceptions.InvalidWorkspaceOperation(
                f"Workspaces cannot have channels in common with incompatible structure: {incompatible_channels}. You can also try a different join operation: {Workspace.valid_joins}."
            )
    return joined_channels


def _join_observations(join, left_observations, right_observations):
    """
    Join two workspace observation specifications.

    Raises:
      ~pyhf.exceptions.InvalidWorkspaceOperation: Observation specifications are incompatible.

    Args:
        join (str): The join operation to apply. See ~pyhf.workspace.Workspace for valid join operations.
        left_observations (list): The left observation specification.
        right_observations (list): The right observation specification.

    Returns:
        :obj:`list`: A joined list of observations. Each observation follows the :obj:`defs.json#observation` `schema <https://scikit-hep.org/pyhf/likelihood.html#observations>`__

    """
    joined_observations = _join_items(join, left_observations, right_observations)
    if join == 'none':
        common_observations = set(c['name'] for c in left_observations).intersection(
            c['name'] for c in right_observations
        )
        if common_observations:
            raise exceptions.InvalidWorkspaceOperation(
                f"Workspaces cannot have any observations in common with the same name: {common_observations}. You can also try a different join operation: {Workspace.valid_joins}."
            )

    elif join == 'outer':
        counted_observations = collections.Counter(
            observation['name'] for observation in joined_observations
        )
        incompatible_observations = [
            observation
            for observation, count in counted_observations.items()
            if count > 1
        ]
        if incompatible_observations:
            raise exceptions.InvalidWorkspaceOperation(
                f"Workspaces cannot have observations in common with incompatible structure: {incompatible_observations}. You can also try a different join operation: {Workspace.valid_joins}."
            )
    return joined_observations


def _join_parameter_configs(measurement_name, join, left_parameters, right_parameters):
    """
    Join two measurement parameter config specifications.

    Raises:
      ~pyhf.exceptions.InvalidWorkspaceOperation: Parameter configuration specifications are incompatible.

    Args:
        measurement_name (str): The name of the measurement being joined (a detail for raising exceptions correctly)
        join (str): The join operation to apply. See ~pyhf.workspace.Workspace for valid join operations.
        left_parameters (list): The left parameter configuration specification.
        right_parameters (list): The right parameter configuration specification.

    Returns:
        :obj:`list`: A joined list of parameter configurations. Each parameter configuration follows the :obj:`defs.json#config` schema

    """
    joined_parameter_configs = _join_items(join, left_parameters, right_parameters)
    if join == 'none':
        common_parameter_configs = set(p['name'] for p in left_parameters).intersection(
            p['name'] for p in right_parameters
        )
        if common_parameter_configs:
            raise exceptions.InvalidWorkspaceOperation(
                f"Workspaces cannot have a measurement ({measurement_name}) specifying different configs for the same parameter: {common_parameter_configs}. You can also try a different join operation: {Workspace.valid_joins}."
            )

    elif join == 'outer':
        counted_parameter_configs = collections.Counter(
            parameter['name'] for parameter in joined_parameter_configs
        )
        incompatible_parameter_configs = [
            parameter
            for parameter, count in counted_parameter_configs.items()
            if count > 1
        ]
        if incompatible_parameter_configs:
            raise exceptions.InvalidWorkspaceOperation(
                f"Workspaces cannot have a measurement ({measurement_name}) with incompatible parameter configs: {incompatible_parameter_configs}. You can also try a different join operation: {Workspace.valid_joins}."
            )
    return joined_parameter_configs


def _join_measurements(join, left_measurements, right_measurements):
    """
    Join two workspace measurement specifications.

    Raises:
      ~pyhf.exceptions.InvalidWorkspaceOperation: Measurement specifications are incompatible.

    Args:
        join (str): The join operation to apply. See ~pyhf.workspace.Workspace for valid join operations.
        left_measurements (list): The left measurement specification.
        right_measurements (list): The right measurement specification.

    Returns:
        :obj:`list`: A joined list of measurements. Each measurement follows the :obj:`defs.json#measurement` `schema <https://scikit-hep.org/pyhf/likelihood.html#measurements>`__

    """
    joined_measurements = _join_items(join, left_measurements, right_measurements)
    if join == 'none':
        common_measurements = set(c['name'] for c in left_measurements).intersection(
            c['name'] for c in right_measurements
        )
        if common_measurements:
            raise exceptions.InvalidWorkspaceOperation(
                f"Workspaces cannot have any measurements in common with the same name: {common_measurements}. You can also try a different join operation: {Workspace.valid_joins}."
            )

    elif join == 'outer':
        # need to store a mapping of measurement name to all measurement objects with that name
        _measurement_mapping = {}
        for measurement in joined_measurements:
            _measurement_mapping.setdefault(measurement['name'], []).append(measurement)
        # first check for incompatible POI
        # then merge parameter configs
        incompatible_poi = [
            measurement_name
            for measurement_name, measurements in _measurement_mapping.items()
            if len(set(measurement['config']['poi'] for measurement in measurements))
            > 1
        ]
        if incompatible_poi:
            raise exceptions.InvalidWorkspaceOperation(
                f"Workspaces cannot have the same measurements with incompatible POI: {incompatible_poi}."
            )

        joined_measurements = []
        for measurement_name, measurements in _measurement_mapping.items():
            if len(measurements) != 1:
                new_measurement = {
                    'name': measurement_name,
                    'config': {
                        'poi': measurements[0]['config']['poi'],
                        'parameters': _join_parameter_configs(
                            measurement_name,
                            join,
                            *[
                                measurement['config']['parameters']
                                for measurement in measurements
                            ],
                        ),
                    },
                }
            else:
                new_measurement = measurements[0]
            joined_measurements.append(new_measurement)
    return joined_measurements


class Workspace(_ChannelSummaryMixin, dict):
    """
    A JSON-serializable object that is built from an object that follows the :obj:`workspace.json` `schema <https://scikit-hep.org/pyhf/likelihood.html#workspace>`__.
    """

    valid_joins = ['none', 'outer', 'left outer', 'right outer']

    def __init__(self, spec, **config_kwargs):
        """Workspaces hold the model, data and measurements."""
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
        """Equality is defined as equal dict representations."""
        if not isinstance(other, Workspace):
            return False
        return dict(self) == dict(other)

    def __ne__(self, other):
        """Negation of equality."""
        return not self == other

    def __repr__(self):
        """Representation of the Workspace."""
        return object.__repr__(self)

    # NB: this is a wrapper function to validate the returned measurement object against the spec
    def get_measurement(self, **config_kwargs):
        """
        Get (or create) a measurement object.

        The following logic is used:

          1. if the poi name is given, create a measurement object for that poi
          2. if the measurement name is given, find the measurement for the given name
          3. if the measurement index is given, return the measurement at that index
          4. if there are measurements but none of the above have been specified, return the 0th measurement

        Raises:
          ~pyhf.exceptions.InvalidMeasurement: If the measurement was not found

        Args:
            poi_name (str): The name of the parameter of interest to create a new measurement from
            measurement_name (str): The name of the measurement to use
            measurement_index (int): The index of the measurement to use

        Returns:
            :obj:`dict`: A measurement object adhering to the schema defs.json#/definitions/measurement

        """
        m = self._get_measurement(**config_kwargs)
        utils.validate(m, 'measurement.json', self.version)
        return m

    def _get_measurement(self, **config_kwargs):
        """See `Workspace::get_measurement`."""
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
            ~pyhf.pdf.Model: A model object adhering to the schema model.json

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

        Raises:
          KeyError: Invalid or missing channel

        Args:
            model (~pyhf.pdf.Model): A model object adhering to the schema model.json
            with_aux (bool): Whether to include auxiliary data from the model or not

        Returns:
            :obj:`list`: data

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

    def _prune_and_rename(
        self,
        prune_modifiers=[],
        prune_modifier_types=[],
        prune_samples=[],
        prune_channels=[],
        prune_measurements=[],
        rename_modifiers={},
        rename_samples={},
        rename_channels={},
        rename_measurements={},
    ):
        """
        Return a new, pruned, renamed workspace specification. This will not modify the original workspace.

        Pruning removes pieces of the workspace whose name or type matches the
        user-provided lists. The pruned, renamed workspace must also be a valid
        workspace.

        A workspace is composed of many named components, such as channels and
        samples, as well as types of systematics (e.g. `histosys`). Components
        can be removed (pruned away) filtering on name or be renamed according
        to the provided :obj:`dict` mapping. Additionally, modifiers of
        specific types can be removed (pruned away).

        This function also handles specific peculiarities, such as
        renaming/removing a channel which needs to rename/remove the
        corresponding `observation`.

        Args:
            prune_modifiers: A :obj:`str` or a :obj:`list` of modifiers to prune.
            prune_modifier_types: A :obj:`str` or a :obj:`list` of modifier types to prune.
            prune_samples: A :obj:`str` or a :obj:`list` of samples to prune.
            prune_channels: A :obj:`str` or a :obj:`list` of channels to prune.
            prune_measurements: A :obj:`str` or a :obj:`list` of measurements to prune.
            rename_modifiers: A :obj:`dict` mapping old modifier name to new modifier name.
            rename_samples: A :obj:`dict` mapping old sample name to new sample name.
            rename_channels: A :obj:`dict` mapping old channel name to new channel name.
            rename_measurements: A :obj:`dict` mapping old measurement name to new measurement name.

        Returns:
            ~pyhf.workspace.Workspace: A new workspace object with the specified components removed or renamed

        """
        newspec = {
            'channels': [
                {
                    'name': rename_channels.get(channel['name'], channel['name']),
                    'samples': [
                        {
                            'name': rename_samples.get(sample['name'], sample['name']),
                            'data': sample['data'],
                            'modifiers': [
                                dict(
                                    modifier,
                                    name=rename_modifiers.get(
                                        modifier['name'], modifier['name']
                                    ),
                                )
                                for modifier in sample['modifiers']
                                if modifier['name'] not in prune_modifiers
                                and modifier['type'] not in prune_modifier_types
                            ],
                        }
                        for sample in channel['samples']
                        if sample['name'] not in prune_samples
                    ],
                }
                for channel in self['channels']
                if channel['name'] not in prune_channels
            ],
            'measurements': [
                {
                    'name': rename_measurements.get(
                        measurement['name'], measurement['name']
                    ),
                    'config': {
                        'parameters': [
                            dict(
                                parameter,
                                name=rename_modifiers.get(
                                    parameter['name'], parameter['name']
                                ),
                            )
                            for parameter in measurement['config']['parameters']
                            if parameter['name'] not in prune_modifiers
                        ],
                        'poi': rename_modifiers.get(
                            measurement['config']['poi'], measurement['config']['poi']
                        ),
                    },
                }
                for measurement in self['measurements']
                if measurement['name'] not in prune_measurements
            ],
            'observations': [
                dict(
                    copy.deepcopy(observation),
                    name=rename_channels.get(observation['name'], observation['name']),
                )
                for observation in self['observations']
                if observation['name'] not in prune_channels
            ],
            'version': self['version'],
        }
        return Workspace(newspec)

    def prune(
        self, modifiers=[], modifier_types=[], samples=[], channels=[], measurements=[]
    ):
        """
        Return a new, pruned workspace specification. This will not modify the original workspace.

        The pruned workspace must also be a valid workspace.

        Args:
            modifiers: A :obj:`str` or a :obj:`list` of modifiers to prune.
            modifier_types: A :obj:`str` or a :obj:`list` of modifier types to prune.
            samples: A :obj:`str` or a :obj:`list` of samples to prune.
            channels: A :obj:`str` or a :obj:`list` of channels to prune.
            measurements: A :obj:`str` or a :obj:`list` of measurements to prune.

        Returns:
            ~pyhf.workspace.Workspace: A new workspace object with the specified components removed

        """
        return self._prune_and_rename(
            prune_modifiers=modifiers,
            prune_modifier_types=modifier_types,
            prune_samples=samples,
            prune_channels=channels,
            prune_measurements=measurements,
        )

    def rename(self, modifiers={}, samples={}, channels={}, measurements={}):
        """
        Return a new workspace specification with certain elements renamed.

        This will not modify the original workspace.
        The renamed workspace must also be a valid workspace.

        Args:
            modifiers: A :obj:`dict` mapping old modifier name to new modifier name.
            samples: A :obj:`dict` mapping old sample name to new sample name.
            channels: A :obj:`dict` mapping old channel name to new channel name.
            measurements: A :obj:`dict` mapping old measurement name to new measurement name.

        Returns:
            ~pyhf.workspace.Workspace: A new workspace object with the specified components renamed

        """
        return self._prune_and_rename(
            rename_modifiers=modifiers,
            rename_samples=samples,
            rename_channels=channels,
            rename_measurements=measurements,
        )

    @classmethod
    def combine(cls, left, right, join='none'):
        """
        Return a new workspace specification that is the combination of the two workspaces.

        The new workspace must also be a valid workspace. A combination of
        workspaces is done by combining the set of:

          - channels,
          - observations, and
          - measurements

        between the two workspaces. If the two workspaces have modifiers that
        follow the same naming convention, then correlations across the two
        workspaces may be possible. In particular, the `lumi` modifier will be
        fully-correlated.

        If the two workspaces have the same measurement (with the same POI),
        those measurements will get merged.

        Raises:
          ~pyhf.exceptions.InvalidWorkspaceOperation: The workspaces have common channel names, incompatible measurements, or incompatible schema versions.

        Args:
            left (~pyhf.workspace.Workspace): A workspace
            right (~pyhf.workspace.Workspace): Another workspace
            join (:obj:`str`): How to join the two workspaces. Pick from "none", "outer", "left outer", or "right outer".

        Returns:
            ~pyhf.workspace.Workspace: A new combined workspace object

        """
        if join not in Workspace.valid_joins:
            raise ValueError(
                f"Workspaces must be joined using one of the valid join operations ({Workspace.valid_joins}); not {join}"
            )
        if join in ['left outer', 'right outer']:
            log.warning(
                "You are using an unsafe join operation. This will silence exceptions that might be raised during a normal 'outer' operation."
            )

        new_version = _join_versions(join, left['version'], right['version'])
        new_channels = _join_channels(join, left['channels'], right['channels'])
        new_observations = _join_observations(
            join, left['observations'], right['observations']
        )
        new_measurements = _join_measurements(
            join, left['measurements'], right['measurements']
        )

        newspec = {
            'channels': new_channels,
            'measurements': new_measurements,
            'observations': new_observations,
            'version': new_version,
        }
        return Workspace(newspec)
