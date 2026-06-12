from __future__ import annotations

import numbers
from collections.abc import Mapping
from pathlib import Path

import jsonschema
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT6

import pyhf.exceptions
from pyhf import tensor
from pyhf.schema import variables
from pyhf.schema.loader import load_schema, read_schema


def _is_array_or_tensor(_checker, instance):
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


def _retrieve_schema(uri: str) -> Resource:
    """
    A ``referencing`` retrieve callback that loads a pyhf schema by its URI.

    Cross-schema ``$ref``\\ s (e.g. ``defs.json``) are resolved against the
    referring schema's ``$id``. For the bundled schemas these are absolute URIs
    under :data:`pyhf.schema.variables.SCHEMA_BASE`; for custom schemas (see
    :class:`pyhf.schema.Schema`) they are paths relative to
    :attr:`pyhf.schema.path`. In both cases stripping the base leaves the disk
    path relative to :attr:`pyhf.schema.path`.

    The schema is read directly from disk via :func:`read_schema` so that
    resolving references does not pollute :data:`pyhf.schema.variables.SCHEMA_CACHE`.
    """
    schema_id = uri.removeprefix(variables.SCHEMA_BASE)
    schema = read_schema(schema_id)
    return Resource.from_contents(schema, default_specification=DRAFT6)


def validate(
    spec: Mapping,
    schema_name: str,
    *,
    version: str | None = None,
    allow_tensors: bool = True,
):
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

    schema = load_schema(str(Path(version).joinpath(schema_name)))

    schema_id = schema["$id"]

    # Resolve cross-schema ``$ref``\ s (e.g. ``defs.json#/...``) relative to the
    # canonical schema ``$id``\ s. The ``_retrieve_schema`` callback lazily loads
    # any referenced schema from disk so the active schema search path
    # (see :class:`pyhf.schema.Schema`) is honored.
    registry = Registry(retrieve=_retrieve_schema).with_resource(
        uri=schema_id,
        resource=Resource.from_contents(schema, default_specification=DRAFT6),
    )

    Validator = jsonschema.Draft6Validator

    if allow_tensors:
        type_checker = Validator.TYPE_CHECKER.redefine(
            "array", _is_array_or_tensor
        ).redefine("number", _is_number_or_tensor_subtype)
        Validator = jsonschema.validators.extend(Validator, type_checker=type_checker)

    validator = Validator(schema, registry=registry, format_checker=None)

    # The pyhf schemas carry a top-level ``$ref`` (e.g. ``model.json`` points at
    # ``defs.json#/definitions/model``). Under draft-06 a sibling ``$ref``
    # suppresses ``$id``, so jsonschema cannot infer the base URI of the root
    # schema and relative references like ``defs.json`` fail to resolve. Anchor
    # the resolver at the schema ``$id`` so those references resolve correctly.
    validator._resolver = registry.resolver(base_uri=schema_id)

    try:
        return validator.validate(spec)
    except jsonschema.ValidationError as err:
        raise pyhf.exceptions.InvalidSpecification(err, schema_name) from err
