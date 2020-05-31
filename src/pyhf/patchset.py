"""
pyhf patchset provides a user-friendly interface for interacting with patchsets.
"""
import logging
import jsonpatch
from . import exceptions
from . import utils
from .workspace import Workspace

log = logging.getLogger(__name__)


class Patch(jsonpatch.JsonPatch):
    """
    A way to store a patch definition as part of a patchset (:class:`~pyhf.patchset.PatchSet`).

    It contains :attr:`~pyhf.patchset.Patch.metadata` about the Patch itself:

      * a descriptive :attr:`~pyhf.patchset.Patch.name`
      * a list of the :attr:`~pyhf.patchset.Patch.values` for each dimension in the phase-space the associated :class:`~pyhf.patchset.PatchSet` is defined for, see :attr:`~pyhf.patchset.PatchSet.labels`

    In addition to the above metadata, the Patch object behaves like the underlying :class:`jsonpatch.JsonPatch`.
    """

    def __init__(self, spec):
        """
        Construct a Patch.

        Args:
            spec (`jsonable`): The patch JSON specification

        Returns:
            patch (:class:`~pyhf.patchset.Patch`): The Patch instance.

        """
        super(Patch, self).__init__(spec['patch'])
        self._metadata = spec['metadata']

    @property
    def metadata(self):
        """ The metadata of the patch """
        return self._metadata

    @property
    def name(self):
        """ The name of the patch """
        return self.metadata['name']

    @property
    def values(self):
        """ The values of the associated labels for the patch """
        return tuple(self.metadata['values'])

    def __repr__(self):
        """ Representation of the object """
        module = type(self).__module__
        qualname = type(self).__qualname__
        return f"<{module}.{qualname} object '{self.name}{self.values}' at {hex(id(self))}>"

    def __eq__(self, other):
        """ Equality for subclass with new attributes """
        if not isinstance(other, Patch):
            return False
        return (
            jsonpatch.JsonPatch.__eq__(self, other) and self.metadata == other.metadata
        )


class PatchSet(object):
    """
    A way to store a collection of patches (:class:`~pyhf.patchset.Patch`).

    It contains :attr:`~PatchSet.metadata` about the PatchSet itself:

      * a high-level :attr:`~pyhf.patchset.PatchSet.description` of what the patches represent or the analysis it is for
      * a list of :attr:`~pyhf.patchset.PatchSet.references` where the patchset is sourced from (e.g. hepdata)
      * a list of :attr:`~pyhf.patchset.PatchSet.digests` corresponding to the background-only workspace the patchset was made for
      * the :attr:`~pyhf.patchset.PatchSet.labels` of the dimensions of the phase-space for what the patches cover

    In addition to the above metadata, the PatchSet object behaves like a:

      * smart list allowing you to iterate over all the patches defined
      * smart dictionary allowing you to access a patch by the patch name or the patch values

    The below example shows various ways one can interact with a :class:`PatchSet` object.

    Example:
        >>> import pyhf
        >>> patchset = pyhf.PatchSet({
        ...     "metadata": {
        ...         "references": { "hepdata": "ins1234567" },
        ...         "description": "example patchset",
        ...         "digests": { "md5": "098f6bcd4621d373cade4e832627b4f6" },
        ...         "labels": ["x", "y"]
        ...     },
        ...     "patches": [
        ...         {
        ...             "metadata": {
        ...                 "name": "patch_name_for_2100x_800y",
        ...                 "values": [2100, 800]
        ...             },
        ...             "patch": [
        ...                 {
        ...                     "op": "add",
        ...                     "path": "/foo/0/bar",
        ...                     "value": {
        ...                         "foo": [1.0]
        ...                     }
        ...                 }
        ...             ]
        ...         }
        ...     ],
        ...     "version": "1.0.0"
        ... })
        ...
        >>> patchset.version
        '1.0.0'
        >>> patchset.references
        {'hepdata': 'ins1234567'}
        >>> patchset.description
        'example patchset'
        >>> patchset.digests
        {'md5': '098f6bcd4621d373cade4e832627b4f6'}
        >>> patchset.labels
        ['x', 'y']
        >>> patchset.patches
        [<pyhf.patchset.Patch object 'patch_name_for_2100x_800y(2100, 800)' at 0x...>]
        >>> patchset['patch_name_for_2100x_800y']
        <pyhf.patchset.Patch object 'patch_name_for_2100x_800y(2100, 800)' at 0x...>
        >>> patchset[(2100,800)]
        <pyhf.patchset.Patch object 'patch_name_for_2100x_800y(2100, 800)' at 0x...>
        >>> patchset[[2100,800]]
        <pyhf.patchset.Patch object 'patch_name_for_2100x_800y(2100, 800)' at 0x...>
        >>> patchset[2100,800]
        <pyhf.patchset.Patch object 'patch_name_for_2100x_800y(2100, 800)' at 0x...>
        >>> for patch in patchset:
        ...     print(patch.name)
        ...
        patch_name_for_2100x_800y
        >>> len(patchset)
        1
    """

    def __init__(self, spec, **config_kwargs):
        """
        Construct a PatchSet.

        Args:
            spec (`jsonable`): The patchset JSON specification
            config_kwargs: Possible keyword arguments for the patchset validation

        Returns:
            patchset (:class:`~pyhf.patchset.PatchSet`): The PatchSet instance.

        """
        self.schema = config_kwargs.pop('schema', 'patchset.json')
        self._version = config_kwargs.pop('version', spec.get('version', None))

        # run jsonschema validation of input specification against the (provided) schema
        log.info(f"Validating spec against schema: {self.schema}")
        utils.validate(spec, self.schema, version=self._version)

        # set properties based on metadata
        self._metadata = spec['metadata']

        # list of all patch objects
        self._patches = []
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
            self._patches.append(patch)
            # register lookup keys for the patch
            self._patches_by_key[patch.name] = patch
            self._patches_by_key[patch.values] = patch

    @property
    def version(self):
        """ The version of the PatchSet """
        return self._version

    @property
    def metadata(self):
        """ The metadata of the PatchSet """
        return self._metadata

    @property
    def references(self):
        """ The references in the PatchSet metadata """
        return self.metadata['references']

    @property
    def description(self):
        """ The description in the PatchSet metadata """
        return self.metadata['description']

    @property
    def digests(self):
        """ The digests in the PatchSet metadata """
        return self.metadata['digests']

    @property
    def labels(self):
        """ The labels in the PatchSet metadata """
        return self.metadata['labels']

    @property
    def patches(self):
        """ The patches in the PatchSet """
        return self._patches

    def __repr__(self):
        """ Representation of the object """
        module = type(self).__module__
        qualname = type(self).__qualname__
        return f"<{module}.{qualname} object with {len(self.patches)} patch{'es' if len(self.patches) != 1 else ''} at {hex(id(self))}>"

    def __getitem__(self, key):
        """
        Access the patch in the patchset by the specified key, either by name or by values.

        Raises:
            ~pyhf.exceptions.InvalidPatchLookup: if the provided patch name is not in the patchset

        Returns:
            patch (:class:`~pyhf.patchset.Patch`): The patch associated with the specified key
        """
        # might be specified as a list, convert to hashable tuple instead for lookup
        if isinstance(key, list):
            key = tuple(key)
        try:
            return self._patches_by_key[key]
        except KeyError:
            raise exceptions.InvalidPatchLookup(
                f'No patch associated with "{key}" is defined in patchset.'
            )

    def __iter__(self):
        """
        Iterate over the defined patches in the patchset.

        Returns:
            iterable (:obj:`iter`): An iterable over the list of patches in the patchset.
        """
        return iter(self.patches)

    def __len__(self):
        """
        The number of patches in the patchset.

        Returns:
            quantity (:obj:`int`): The number of patches in the patchset.
        """
        return len(self.patches)

    def verify(self, spec):
        """
        Verify the patchset digests against a background-only workspace specification. Verified if no exception was raised.

        Args:
            spec (:class:`~pyhf.workspace.Workspace`): The workspace specification to verify the patchset against.

        Raises:
            ~pyhf.exceptions.PatchSetVerificationError: if the patchset cannot be verified against the workspace specification

        Returns:
            None
        """
        for hash_alg, digest in self.digests.items():
            digest_calc = utils.digest(spec, algorithm=hash_alg)
            if not digest_calc == digest:
                raise exceptions.PatchSetVerificationError(
                    f"The digest verification failed for hash algorithm '{hash_alg}'. Expected: {digest}. Got: {digest_calc}"
                )

    def apply(self, spec, key):
        """
        Apply the patch associated with the key to the background-only workspace specificatiom.

        Args:
            spec (:class:`~pyhf.workspace.Workspace`): The workspace specification to verify the patchset against.
            key (:obj:`str` or :obj:`tuple` of :obj:`int`/:obj:`float`): The key to look up the associated patch - either a name or a set of values.

        Raises:
            ~pyhf.exceptions.InvalidPatchLookup: if the provided patch name is not in the patchset
            ~pyhf.exceptions.PatchSetVerificationError: if the patchset cannot be verified against the workspace specification

        Returns:
            workspace (:class:`~pyhf.workspace.Workspace`): The background-only workspace with the patch applied.
        """
        self.verify(spec)
        return Workspace(self[key].apply(spec))
