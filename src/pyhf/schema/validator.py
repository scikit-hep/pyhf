import numbers
from pathlib import Path
import jsonschema
import logging
import pyhf.exceptions
from pyhf import tensor
from pyhf.schema import variables
from pyhf.schema.loader import load_schema
from pyhf.typing import Workspace, Model, Measurement, PatchSet

log = logging.getLogger(__name__)


def _is_array_or_tensor(checker, instance):
    """
    A helper function for allowing the validation of tensors as list types in schema validation.

    .. warning:

        This will check for valid array types using any backends that have been loaded so far.
    """
    return isinstance(instance, (list, *tensor.array_types))


def _is_number_or_tensor_subtype(checker, instance):
    """
    A helper function for allowing the validation of tensor contents as number types in schema validation.

    .. warning:
        This will check for valid array subtypes using any backends that have been loaded so far.
    """
    is_number = jsonschema._types.is_number(checker, instance)
    if is_number:
        return True
    return isinstance(instance, (numbers.Number, *tensor.array_subtypes))


def validate(
    spec: Workspace | Model | Measurement | PatchSet,
    schema_name: str,
    *,
    version: str | None = None,
    allow_tensors: bool = True,
) -> None:
    """
    Validate the provided instance, ``spec``, against the schema associated with ``schema_name``.

    Args:
        spec (:obj:`object`): An object instance to validate against a schema.
        schema_name (:obj:`string`): The name of a schema to validate against.
         See :func:`pyhf.schema.load_schema` for more details.
        version (:obj:`string`): The version of the schema to use.
         See :func:`pyhf.schema.load_schema` for more details.
        allow_tensors (:obj:`bool`): A flag to enable or disable tensors as part of schema validation.
         If enabled, tensors in the ``spec`` will be treated like python :obj:`list`.
         Default: ``True``.

    Raises:
        ~pyhf.exceptions.InvalidSpecification: if the provided instance does not validate against the schema.

    Returns:
        None: if there are no errors with the provided instance.

    Example:
        >>> import pyhf
        >>> model = pyhf.simplemodels.uncorrelated_background(
        ...     signal=[12.0, 11.0], bkg=[50.0, 52.0], bkg_uncertainty=[3.0, 7.0]
        ... )
        >>> pyhf.schema.validate(model.spec, "model.json")
        >>>
    """

    version = version or variables.SCHEMA_VERSION
    if version != variables.SCHEMA_VERSION:
        log.warning(
            f"Specification requested version {version} but latest is {variables.SCHEMA_VERSION}. Upgrade your specification or downgrade pyhf."
        )

    schema = load_schema(str(Path(version).joinpath(schema_name)))

    # note: trailing slash needed for RefResolver to resolve correctly and by
    # design, pathlib strips trailing slashes. See ref below:
    # * https://bugs.python.org/issue21039
    # * https://github.com/python/cpython/issues/65238
    resolver = jsonschema.RefResolver(
        base_uri=f"{Path(variables.schemas).joinpath(version).as_uri()}/",
        referrer=schema_name,
        store=variables.SCHEMA_CACHE,
    )

    Validator = jsonschema.Draft202012Validator

    if allow_tensors:
        type_checker = Validator.TYPE_CHECKER.redefine(
            "array", _is_array_or_tensor
        ).redefine("number", _is_number_or_tensor_subtype)
        Validator = jsonschema.validators.extend(Validator, type_checker=type_checker)

    validator = Validator(schema, resolver=resolver, format_checker=None)

    try:
        return validator.validate(spec)
    except jsonschema.ValidationError as err:
        raise pyhf.exceptions.InvalidSpecification(err, schema_name)  # type: ignore[no-untyped-call]
