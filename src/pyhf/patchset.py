"""
pyhf patchset provides a user-friendly interface for interacting with patchsets.
"""
import logging
import jsonpatch
from . import exceptions
from . import utils

log = logging.getLogger(__name__)


class Patch(jsonpatch.JsonPatch):
    """
    A patch.
    """

    def __init__(self, spec):
        """
        Construct a Patch.

        Args:
            spec (`jsonable`): The patch JSON specification

        Returns:
            patch (`Patch`): The Patch instance.

        """
        super(Patch, self).__init__(spec['patch'])
        self.metadata = spec['metadata']

    def __repr__(self):
        """ Representation of the patch object """
        return f"<Patch object '{self.name}{self.values}' at {hex(id(self))}>"

    def __eq__(self, other):
        """ Equality for subclass with new attributes """
        if not isinstance(other, Patch):
            return False
        return (
            jsonpatch.JsonPatch.__eq__(self, other) and self.metadata == other.metadata
        )

    @property
    def name(self):
        return self.metadata['name']

    @property
    def values(self):
        return tuple(self.metadata['values'])


class PatchSet(object):
    """
    A collection of patches.
    """

    def __init__(self, spec, **config_kwargs):
        """
        Construct a PatchSet.

        Args:
            spec (`jsonable`): The patchset JSON specification
            config_kwargs: Possible keyword arguments for the patchset validation

        Returns:
            patchset (`PatchSet`): The PatchSet instance.

        """
        self.schema = config_kwargs.pop('schema', 'patchset.json')
        self.version = config_kwargs.pop('version', spec.get('version', None))

        # run jsonschema validation of input specification against the (provided) schema
        log.info(f"Validating spec against schema: {self.schema}")
        utils.validate(spec, self.schema, version=self.version)

        # set properties based on metadata
        self.metadata = spec['metadata']

        # list of all patch objects
        self.patches = []
        # look-up table for retrieving patch by name or values
        self._patches_by_key = {'name': {}, 'values': {}}

        # inflate all patches
        for patchspec in spec['patches']:
            patch = Patch(patchspec)

            if patch.name in self._patches_by_key:
                raise exceptions.InvalidPatchSet(
                    f'Multiple patches were defined by name for {patch}.'
                )

            if patch.values in self._patches_by_key:
                raise exceptions.InvalidPatchSet(
                    f'Multiple patches were defined by values for {patch}.'
                )

            if len(patch.values) != len(self.labels):
                raise exceptions.InvalidPatchSet(
                    f'Incompatible number of values ({len(patch.values)} for {patch} in patchset. Expected {len(self.labels)}.'
                )

            # all good, register patch
            self.patches.append(patch)
            # register lookup keys for the patch
            self._patches_by_key[patch.name] = patch
            self._patches_by_key[patch.values] = patch

    def __repr__(self):
        """ Representation of the patchset object """
        return f"<PatchSet object with {len(self.patches)} patch{'es' if len(self.patches) != 1 else ''} at {hex(id(self))}>"

    def __getitem__(self, key):
        # might be specified as a list, convert to hashable tuple instead for lookup
        if isinstance(key, list):
            key = tuple(key)
        try:
            return self._patches_by_key[key]
        except KeyError:
            raise exceptions.InvalidPatchLookup(
                f'No patch associated with "{key}" is defined in patchset.'
            )

    # make it iterable
    def __iter__(self):
        return iter(self.patches)

    # give it a length
    def __len__(self):
        return len(self.patches)

    @property
    def references(self):
        return self.metadata['references']

    @property
    def description(self):
        return self.metadata['description']

    @property
    def digests(self):
        return self.metadata['digests']

    @property
    def labels(self):
        return self.metadata['labels']
