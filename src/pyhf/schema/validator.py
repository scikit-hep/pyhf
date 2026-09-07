from __future__ import annotations

import numbers
from collections.abc import Mapping
from urllib.parse import urljoin, urlsplit

import jsonschema
import referencing.exceptions
from referencing import Registry, Resource
from referencing.exceptions import NoSuchResource
from referencing.jsonschema import DRAFT6

import pyhf.exceptions
from pyhf import tensor
from pyhf.schema import variables
from pyhf.schema.loader import load_schema


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

    :func:`validate` resolves the top-level ``$ref`` of a schema (e.g.
    ``defs.json``) against the directory of the requested version under
    :data:`pyhf.schema.variables.SCHEMA_BASE`, so the URIs that reach this
    callback are the bundled ``$id``\\ s. Stripping the base leaves the path
    relative to :attr:`pyhf.schema.path`, which is loaded through
    :func:`pyhf.schema.load_schema` so that referenced schemas are cached in
    :data:`pyhf.schema.variables.SCHEMA_CACHE` after the first load. Nothing is
    ever fetched from the network.

    Raises:
        ~referencing.exceptions.NoSuchResource: if ``uri`` is an absolute URI
         that is not under :data:`pyhf.schema.variables.SCHEMA_BASE`, as
         nothing under :attr:`pyhf.schema.path` can satisfy it.
        ~pyhf.exceptions.SchemaNotFound: if the schema is not found under
         :attr:`pyhf.schema.path`. ``referencing`` surfaces this as
         :class:`~referencing.exceptions.Unresolvable` with the cause chain
         intact.
    """
    parts = urlsplit(uri.removeprefix(variables.SCHEMA_BASE))
    if parts.scheme or parts.netloc:
        raise NoSuchResource(ref=uri)
    return Resource.from_contents(load_schema(parts.path), default_specification=DRAFT6)


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
        ~pyhf.exceptions.SchemaNotFound: if the schema, or a schema it references, cannot be found.

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

    schema = load_schema(f"{version}/{schema_name}")

    Validator = jsonschema.Draft6Validator

    if allow_tensors:
        type_checker = Validator.TYPE_CHECKER.redefine(
            "array", _is_array_or_tensor
        ).redefine("number", _is_number_or_tensor_subtype)
        Validator = jsonschema.validators.extend(Validator, type_checker=type_checker)

    # Every pyhf schema is a bare draft-06 ``$ref`` shell (e.g. ``model.json``
    # points at ``defs.json#/definitions/model``). Under draft-06 a sibling
    # ``$ref`` suppresses ``$id`` (c.f. referencing.jsonschema._legacy_id), so
    # jsonschema cannot infer the base URI of the root schema and the relative
    # ``defs.json`` reference fails to resolve. Enter validation through the
    # absolute form of that reference instead, anchored at the on-disk directory
    # of the requested version rather than at the schema's own ``$id``: as with
    # the RefResolver base_uri this replaces, a stale or copy-pasted ``$id`` then
    # cannot redirect ``defs.json`` to another version. ``_retrieve_schema``
    # loads it from ``pyhf.schema.path``. Draft-06 ignores the siblings of
    # ``$ref``, so this is equivalent to validating against the schema itself.
    # (Referencing the document by its ``$id`` would instead make jsonschema
    # re-select the stock Draft6Validator from the document's ``$schema`` and
    # drop the tensor-aware type checker, c.f. jsonschema.validators.validator_for.)
    base_uri = f"{variables.SCHEMA_BASE}{version}/"
    root = {"$ref": urljoin(base_uri, schema["$ref"])} if "$ref" in schema else schema
    validator = Validator(
        root, registry=Registry(retrieve=_retrieve_schema), format_checker=None
    )

    try:
        return validator.validate(spec)
    except jsonschema.ValidationError as err:
        raise pyhf.exceptions.InvalidSpecification(err, schema_name) from err
    except referencing.exceptions.Unresolvable as err:
        msg = (
            f"Could not resolve the schema reference {err.ref!r} while validating against "
            f"{schema_name} (version {version}). Referenced schemas must have an $id under "
            f"{variables.SCHEMA_BASE} or be relative paths under pyhf.schema.path ({variables.schemas})."
        )
        raise pyhf.exceptions.SchemaNotFound(msg) from err
