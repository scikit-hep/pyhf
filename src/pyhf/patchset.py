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
        # set properties based on metadata
        self.name = spec['metadata']['name']
        self.values = tuple(spec['metadata']['values'])

    def __repr__(self):
        """ Representation of the patch object """
        return f"<Patch object '{self.name}{self.values}' at {hex(id(self))}>"


class Patchset(object):
    """
    A collection of patches.
    """

    def __init__(self, spec, **config_kwargs):
        """
        Construct a Patchset.

        Args:
            spec (`jsonable`): The patchset JSON specification
            config_kwargs: Possible keyword arguments for the patchset validation

        Returns:
            patchset (`Patchset`): The Patchset instance.

        """
        self.schema = config_kwargs.pop('schema', 'patchset.json')
        self.version = config_kwargs.pop('version', spec.get('version', None))

        # run jsonschema validation of input specification against the (provided) schema
        log.info(f"Validating spec against schema: {self.schema}")
        utils.validate(spec, self.schema, version=self.version)

        # set properties based on metadata
        self.__dict__.update(spec['metadata'])

        # list of all patch objects
        self.patches = []
        # look-up table for retrieving patch by name or values
        self._patches_by_key = {'name': {}, 'values': {}}

        # inflate all patches
        for patchspec in spec['patches']:
            patch = Patch(patchspec)

            if patch.name in self._patches_by_key:
                raise exceptions.InvalidPatchset(
                    f'Multiple patches were defined by name for {patch}.'
                )

            if patch.values in self._patches_by_key:
                raise exceptions.InvalidPatchset(
                    f'Multiple patches were defined by values for {patch}.'
                )

            # all good, register patch
            self.patches.append(patch)
            # register lookup keys for the patch
            self._patches_by_key[patch.name] = patch
            self._patches_by_key[patch.values] = patch

    def __repr__(self):
        """ Representation of the patchset object """
        return f"<Patchset object with {len(self.patches)} patch{'es' if len(self.patches) != 1 else ''} at {hex(id(self))}>"

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
