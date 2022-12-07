from pyhf.schema import variables
from pyhf.typing import Workspace, PatchSet, SchemaVersion
import copy


def upgrade_workspace(spec: Workspace, *, to_version: SchemaVersion) -> Workspace:
    """
    Upgrade the provided workspace specification to latest version.

    Args:
        spec (dict): The specification to validate.
        schema_name (str): The name of the schema to upgrade.

    Returns:
        upgraded_spec (dict): Upgraded workspace specification.

    Raises:
        pyhf.exceptions.InvalidSpecification: the specification is invalid
    """

    version = spec['version']
    latest_version = variables.SCHEMA_VERSION['workspace.json']

    if version == latest_version:
        return spec

    new_spec = copy.deepcopy(spec)
    if version == '1.0.0':
        new_spec['version'] = variables.SCHEMA_VERSION['workspace.json']
    return new_spec


def upgrade_patchset(spec: PatchSet) -> PatchSet:
    """
    Upgrade the provided patchset specification to latest version.

    Args:
        spec (dict): The specification to validate.
        schema_name (str): The name of the schema to upgrade.

    Returns:
        upgraded_spec (dict): Upgraded patchset specification.

    Raises:
        pyhf.exceptions.InvalidSpecification: the specification is invalid
    """

    version = spec['version']

    new_spec = copy.deepcopy(spec)
    if version == '1.0.0':
        new_spec['version'] = variables.SCHEMA_VERSION['patchset.json']
    return new_spec
