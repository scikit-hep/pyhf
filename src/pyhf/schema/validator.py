import jsonschema
import pyhf.exceptions
from pyhf.schema.loader import load_schema
from pyhf.schema import variables
from typing import Union


def validate(spec: dict, schema_name: str, version: Union[str, None] = None, allow_tensors: bool = True):
    """
    Validate the provided instance, ``spec``, against the schema associated with ``schema_id``.

    Args:
        spec (:obj:`object`): An object instance to validate against a schema
        schema_id (:obj:`string`): The name of a schema to validate against. See :func:`pyhf.utils.load_schema` for more details.
        version (:obj:`string`): The version of the schema to use. See :func:`pyhf.utils.load_schema` for more details.
        allow_tensors (:obj:`bool`): A flag to enable or disable tensors as part of schema validation. If enabled, tensors in the ``spec`` will be treated like python :obj:`list`. Default: ``True``.

    Raises:
        ~pyhf.exceptions.InvalidSpecification: if the provided instance does not validate against the schema.

    Returns:
        None: if there are no errors with the provided instance

    Example:
        >>> import pyhf
        >>> model = pyhf.simplemodels.uncorrelated_background(
        ...     signal=[12.0, 11.0], bkg=[50.0, 52.0], bkg_uncertainty=[3.0, 7.0]
        ... )
        >>> pyhf.utils.validate(model.spec, 'model.json')
        >>>
    """

    version = version or variables.SCHEMA_VERSION

    schema = load_schema(f'{version}/{schema_name}')

    # note: trailing slash needed for RefResolver to resolve correctly
    resolver = jsonschema.RefResolver(
        base_uri=f"file://{variables.schemas}/",
        referrer=f"{version}/{schema_name}",
        store=variables.SCHEMA_CACHE,
    )
    validator = jsonschema.Draft6Validator(
        schema, resolver=resolver, format_checker=None
    )

    try:
        return validator.validate(spec)
    except jsonschema.ValidationError as err:
        raise pyhf.exceptions.InvalidSpecification(err, schema_name)
