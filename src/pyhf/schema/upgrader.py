from pyhf.schema import variables
from typing import Mapping
import copy


def upgrade_workspace(spec: Mapping) -> Mapping:
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

    new_spec = copy.deepcopy(spec)
    if version == '1.0.0':
        new_spec['version'] = variables.SCHEMA_VERSION
    return spec


def upgrade_patchset(spec: Mapping) -> Mapping:
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
        new_spec['version'] = variables.SCHEMA_VERSION
    return spec
