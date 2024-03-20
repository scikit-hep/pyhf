from __future__ import annotations

from pyhf.typing import Workspace, PatchSet, SchemaVersion, UpgradeProtocol
import copy


class Upgrade_1_0_1:
    """
    Used for testing functionality of upgrade.
    """

    version: SchemaVersion = '1.0.1'

    @classmethod
    def workspace(cls, spec: Workspace) -> Workspace:
        """
        Upgrade the provided workspace specification.

        Args:
            spec (dict): The specification to validate.
            schema_name (str): The name of the schema to upgrade.

        Returns:
            upgraded_spec (dict): Upgraded workspace specification.

        Raises:
            pyhf.exceptions.InvalidSpecification: the specification is invalid
        """

        new_spec = copy.deepcopy(spec)
        if spec['version'] == '1.0.0':
            new_spec['version'] = cls.version
        return new_spec

    @classmethod
    def patchset(cls, spec: PatchSet) -> PatchSet:
        """
        Upgrade the provided patchset specification.

        Args:
            spec (dict): The specification to validate.
            schema_name (str): The name of the schema to upgrade.

        Returns:
            upgraded_spec (dict): Upgraded patchset specification.

        Raises:
            pyhf.exceptions.InvalidSpecification: the specification is invalid
        """

        new_spec = copy.deepcopy(spec)
        if spec['version'] == '1.0.0':
            new_spec['version'] = cls.version
        return new_spec


def upgrade(*, to_version: SchemaVersion | None = None) -> type[UpgradeProtocol]:
    if to_version is None or to_version == '1.0.1':
        return Upgrade_1_0_1

    raise ValueError(f'{to_version} is not a valid version to upgrade to.')
