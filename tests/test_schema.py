import importlib
import json
import sys

import pytest
from pytest_socket import socket_disabled  # noqa: F401

import pyhf


@pytest.mark.parametrize('version', ['1.0.0'])
@pytest.mark.parametrize(
    'schema', ['defs.json', 'measurement.json', 'model.json', 'workspace.json']
)
def test_get_schema(version, schema):
    assert pyhf.schema.load_schema(f'{version}/{schema}')


def test_load_missing_schema():
    with pytest.raises(IOError):
        pyhf.schema.load_schema('fake_schema.json')


def test_schema_attributes():
    assert hasattr(pyhf.schema, 'version')
    assert hasattr(pyhf.schema, 'path')
    assert pyhf.schema.version
    assert pyhf.schema.path


def test_schema_callable():
    assert callable(pyhf.schema)


@pytest.fixture
def self_restoring_schema_globals():
    old_path = pyhf.schema.path
    old_cache = dict(pyhf.schema.variables.SCHEMA_CACHE)
    yield old_path, old_cache
    pyhf.schema(old_path)
    pyhf.schema.variables.SCHEMA_CACHE = old_cache


def test_schema_changeable(datadir, monkeypatch, self_restoring_schema_globals):
    monkeypatch.setattr(
        pyhf.schema.variables, 'schemas', pyhf.schema.variables.schemas, raising=True
    )
    old_path, old_cache = self_restoring_schema_globals
    new_path = datadir / 'customschema'

    with pytest.raises(pyhf.exceptions.SchemaNotFound):
        pyhf.Workspace(json.load(open(datadir / 'customschema' / 'custom.json')))

    pyhf.schema(new_path)
    assert old_path != pyhf.schema.path
    assert new_path == pyhf.schema.path
    assert pyhf.schema.variables.SCHEMA_CACHE is not old_cache
    assert len(pyhf.schema.variables.SCHEMA_CACHE) == 0
    assert pyhf.Workspace(json.load(open(new_path / 'custom.json')))
    assert len(pyhf.schema.variables.SCHEMA_CACHE) == 1


def test_schema_changeable_context(datadir, monkeypatch, self_restoring_schema_globals):
    monkeypatch.setattr(
        pyhf.schema.variables, 'schemas', pyhf.schema.variables.schemas, raising=True
    )
    old_path, old_cache = self_restoring_schema_globals
    new_path = datadir / 'customschema'

    assert old_path == pyhf.schema.path
    with pyhf.schema(new_path):
        assert old_path != pyhf.schema.path
        assert new_path == pyhf.schema.path
        assert pyhf.schema.variables.SCHEMA_CACHE is not old_cache
        assert len(pyhf.schema.variables.SCHEMA_CACHE) == 0
        assert pyhf.Workspace(json.load(open(new_path / 'custom.json')))
        assert len(pyhf.schema.variables.SCHEMA_CACHE) == 1
    assert old_path == pyhf.schema.path
    assert old_cache == pyhf.schema.variables.SCHEMA_CACHE


def test_schema_changeable_context_error(
    datadir, monkeypatch, self_restoring_schema_globals
):
    monkeypatch.setattr(
        pyhf.schema.variables, 'schemas', pyhf.schema.variables.schemas, raising=True
    )
    old_path, old_cache = self_restoring_schema_globals
    new_path = datadir / 'customschema'

    with pytest.raises(ZeroDivisionError):
        with pyhf.schema(new_path):
            # this populates the current cache
            pyhf.Workspace(json.load(open(new_path / 'custom.json')))
            raise ZeroDivisionError()
    assert old_path == pyhf.schema.path
    assert old_cache == pyhf.schema.variables.SCHEMA_CACHE


def test_no_channels():
    spec = {'channels': []}
    with pytest.raises(pyhf.exceptions.InvalidSpecification):
        pyhf.Model(spec)


def test_no_samples():
    spec = {'channels': [{'name': 'channel', 'samples': []}]}
    with pytest.raises(pyhf.exceptions.InvalidSpecification):
        pyhf.Model(spec)


def test_sample_missing_data():
    spec = {
        'channels': [
            {
                'name': 'channel',
                'samples': [{'name': 'sample', 'data': [], 'modifiers': []}],
            }
        ]
    }
    with pytest.raises(pyhf.exceptions.InvalidSpecification):
        pyhf.Model(spec)


def test_sample_missing_name():
    spec = {
        'channels': [{'name': 'channel', 'samples': [{'data': [1], 'modifiers': []}]}]
    }
    with pytest.raises(pyhf.exceptions.InvalidSpecification):
        pyhf.Model(spec)


def test_sample_missing_all_modifiers():
    spec = {
        'channels': [
            {
                'name': 'channel',
                'samples': [{'name': 'sample', 'data': [10.0], 'modifiers': []}],
            }
        ]
    }
    with pytest.raises(pyhf.exceptions.InvalidModel):
        pyhf.Model(spec)


def test_one_sample_missing_modifiers():
    spec = {
        'channels': [
            {
                'name': 'channel',
                'samples': [
                    {'name': 'sample', 'data': [10.0], 'modifiers': []},
                    {
                        'name': 'another_sample',
                        'data': [5.0],
                        'modifiers': [
                            {'name': 'mypoi', 'type': 'normfactor', 'data': None}
                        ],
                    },
                ],
            }
        ]
    }
    pyhf.Model(spec, poi_name='mypoi')


def test_add_unknown_modifier():
    spec = {
        'channels': [
            {
                'name': 'channel',
                'samples': [
                    {
                        'name': 'ttbar',
                        'data': [1],
                        'modifiers': [
                            {
                                'name': 'a_name',
                                'type': 'this_should_not_exist',
                                'data': [1],
                            }
                        ],
                    }
                ],
            }
        ]
    }
    with pytest.raises(pyhf.exceptions.InvalidSpecification):
        pyhf.Model(spec)


def test_empty_staterror():
    spec = {
        'channels': [
            {
                'name': 'channel',
                'samples': [
                    {
                        'name': 'sample',
                        'data': [10.0],
                        'modifiers': [
                            {
                                'name': 'staterror_channel',
                                'type': 'staterror',
                                'data': [],
                            }
                        ],
                    }
                ],
            }
        ]
    }
    with pytest.raises(pyhf.exceptions.InvalidSpecification):
        pyhf.Model(spec)


def test_empty_shapesys():
    spec = {
        'channels': [
            {
                'name': 'channel',
                'samples': [
                    {
                        'name': 'sample',
                        'data': [10.0],
                        'modifiers': [
                            {'name': 'sample_norm', 'type': 'shapesys', 'data': []}
                        ],
                    }
                ],
            }
        ]
    }
    with pytest.raises(pyhf.exceptions.InvalidSpecification):
        pyhf.Model(spec)


def test_empty_histosys():
    spec = {
        'channels': [
            {
                'name': 'channel',
                'samples': [
                    {
                        'name': 'sample',
                        'data': [10.0],
                        'modifiers': [
                            {
                                'name': 'modifier',
                                'type': 'histosys',
                                'data': {'lo_data': [], 'hi_data': []},
                            }
                        ],
                    }
                ],
            }
        ]
    }
    with pytest.raises(pyhf.exceptions.InvalidSpecification):
        pyhf.Model(spec)


def test_additional_properties():
    spec = {
        'channels': [
            {
                'name': 'channel',
                'samples': [
                    {'name': 'sample', 'data': [10.0], 'modifiers': []},
                    {
                        'name': 'another_sample',
                        'data': [5.0],
                        'modifiers': [
                            {'name': 'mypoi', 'type': 'normfactor', 'data': None}
                        ],
                    },
                ],
            }
        ],
        'fake_additional_property': 2,
    }
    with pytest.raises(pyhf.exceptions.InvalidSpecification):
        pyhf.Model(spec)


def test_parameters_definition():
    spec = {
        'channels': [
            {
                'name': 'channel',
                'samples': [
                    {'name': 'sample', 'data': [10.0], 'modifiers': []},
                    {
                        'name': 'another_sample',
                        'data': [5.0],
                        'modifiers': [
                            {'name': 'mypoi', 'type': 'normfactor', 'data': None}
                        ],
                    },
                ],
            }
        ],
        'parameters': [{'name': 'mypoi'}],
    }
    pyhf.Model(spec, poi_name='mypoi')


def test_parameters_incorrect_format():
    spec = {
        'channels': [
            {
                'name': 'channel',
                'samples': [
                    {'name': 'sample', 'data': [10.0], 'modifiers': []},
                    {
                        'name': 'another_sample',
                        'data': [5.0],
                        'modifiers': [
                            {'name': 'mypoi', 'type': 'normfactor', 'data': None}
                        ],
                    },
                ],
            }
        ],
        'parameters': {'a': 'fake', 'object': 2},
    }
    with pytest.raises(pyhf.exceptions.InvalidSpecification):
        pyhf.Model(spec, poi_name='mypoi')


def test_parameters_duplicated():
    spec = {
        'channels': [
            {
                'name': 'channel',
                'samples': [
                    {'name': 'sample', 'data': [10.0], 'modifiers': []},
                    {
                        'name': 'another_sample',
                        'data': [5.0],
                        'modifiers': [
                            {'name': 'mypoi', 'type': 'normfactor', 'data': None}
                        ],
                    },
                ],
            }
        ],
        'parameters': [{'name': 'mypoi'}, {'name': 'mypoi'}],
    }
    with pytest.raises(pyhf.exceptions.InvalidModel):
        pyhf.Model(spec, poi_name='mypoi')


def test_parameters_fixed():
    spec = {
        'channels': [
            {
                'name': 'channel',
                'samples': [
                    {
                        'name': 'sample',
                        'data': [10.0],
                        'modifiers': [
                            {'name': 'unfixed', 'type': 'normfactor', 'data': None}
                        ],
                    },
                    {
                        'name': 'another_sample',
                        'data': [5.0],
                        'modifiers': [
                            {'name': 'mypoi', 'type': 'normfactor', 'data': None}
                        ],
                    },
                ],
            }
        ],
        'parameters': [{'name': 'mypoi', 'inits': [1], 'fixed': True}],
    }
    pyhf.Model(spec, poi_name='mypoi')


def test_parameters_all_props():
    spec = {
        'channels': [
            {
                'name': 'channel',
                'samples': [
                    {'name': 'sample', 'data': [10.0], 'modifiers': []},
                    {
                        'name': 'another_sample',
                        'data': [5.0],
                        'modifiers': [
                            {'name': 'mypoi', 'type': 'normfactor', 'data': None}
                        ],
                    },
                ],
            }
        ],
        'parameters': [{'name': 'mypoi', 'inits': [1], 'bounds': [[0, 1]]}],
    }
    pyhf.Model(spec, poi_name='mypoi')


@pytest.mark.parametrize(
    'bad_parameter',
    [
        {'name': 'mypoi', 'inits': ['a']},
        {'name': 'mypoi', 'bounds': [0, 1]},
        {'name': 'mypoi', 'auxdata': ['a']},
        {'name': 'mypoi', 'factors': ['a']},
        {'name': 'mypoi', 'paramset_type': 'fake_paramset_type'},
        {'name': 'mypoi', 'n_parameters': 5},
        {'name': 'mypoi', 'op_code': 'fake_op_code'},
    ],
    ids=[
        'inits',
        'bounds',
        'auxdata',
        'factors',
        'paramset_type',
        'n_parameters',
        'op_code',
    ],
)
def test_parameters_bad_parameter(bad_parameter):
    spec = {
        'channels': [
            {
                'name': 'channel',
                'samples': [
                    {'name': 'sample', 'data': [10.0], 'modifiers': []},
                    {
                        'name': 'another_sample',
                        'data': [5.0],
                        'modifiers': [
                            {'name': 'mypoi', 'type': 'normfactor', 'data': None}
                        ],
                    },
                ],
            }
        ],
        'parameters': [bad_parameter],
    }
    with pytest.raises(pyhf.exceptions.InvalidSpecification):
        pyhf.Model(spec, poi_name='mypoi')


@pytest.mark.parametrize(
    'bad_parameter', [{'name': 'mypoi', 'factors': [0.0]}], ids=['factors']
)
def test_parameters_normfactor_bad_attribute(bad_parameter):
    spec = {
        'channels': [
            {
                'name': 'channel',
                'samples': [
                    {'name': 'sample', 'data': [10.0], 'modifiers': []},
                    {
                        'name': 'another_sample',
                        'data': [5.0],
                        'modifiers': [
                            {'name': 'mypoi', 'type': 'normfactor', 'data': None}
                        ],
                    },
                ],
            }
        ],
        'parameters': [bad_parameter],
    }
    with pytest.raises(pyhf.exceptions.InvalidModel):
        pyhf.Model(spec, poi_name='mypoi')


def test_histosys_additional_properties():
    spec = {
        'channels': [
            {
                'name': 'channel',
                'samples': [
                    {
                        'name': 'sample',
                        'data': [10.0],
                        'modifiers': [
                            {
                                'name': 'histosys',
                                'type': 'histosys',
                                'data': {
                                    'hi_data': [1.0],
                                    'lo_data': [0.5],
                                    'foo': 2.0,
                                },
                            }
                        ],
                    }
                ],
            }
        ]
    }
    with pytest.raises(pyhf.exceptions.InvalidSpecification):
        pyhf.Model(spec)


def test_normsys_additional_properties():
    spec = {
        'channels': [
            {
                'name': 'channel',
                'samples': [
                    {
                        'name': 'sample',
                        'data': [10.0],
                        'modifiers': [
                            {
                                'name': 'normsys',
                                'type': 'normsys',
                                'data': {'hi': 1.0, 'lo': 0.5, 'foo': 2.0},
                            }
                        ],
                    }
                ],
            }
        ]
    }
    with pytest.raises(pyhf.exceptions.InvalidSpecification):
        pyhf.Model(spec)


@pytest.mark.parametrize(
    'patch',
    [
        {"op": "add", "path": "/foo/0/bar", "value": {"foo": [1.0]}},
        {"op": "replace", "path": "/foo/0/bar", "value": {"foo": [1.0]}},
        {"op": "test", "path": "/foo/0/bar", "value": {"foo": [1.0]}},
        {"op": "remove", "path": "/foo/0/bar"},
        {"op": "move", "path": "/foo/0/bar", "from": "/foo/0/baz"},
        {"op": "copy", "path": "/foo/0/bar", "from": "/foo/0/baz"},
    ],
    ids=['add', 'replace', 'test', 'remove', 'move', 'copy'],
)
def test_jsonpatch(patch):
    pyhf.schema.validate([patch], 'jsonpatch.json')


@pytest.mark.parametrize(
    'patch',
    [
        {"path": "/foo/0/bar"},
        {"op": "add", "path": "/foo/0/bar", "from": {"foo": [1.0]}},
        {"op": "add", "path": "/foo/0/bar"},
        {"op": "add", "value": {"foo": [1.0]}},
        {"op": "remove"},
        {"op": "move", "path": "/foo/0/bar"},
        {"op": "move", "from": "/foo/0/baz"},
    ],
    ids=[
        'noop',
        'add_from_novalue',
        'add_novalue',
        'add_nopath',
        'remove_nopath',
        'move_nofrom',
        'move_nopath',
    ],
)
def test_jsonpatch_fail(patch):
    with pytest.raises(pyhf.exceptions.InvalidSpecification):
        pyhf.schema.validate([patch], 'jsonpatch.json')


@pytest.mark.parametrize('patchset_file', ['patchset_good.json'])
def test_patchset(datadir, patchset_file):
    patchset = json.load(open(datadir.joinpath(patchset_file)))
    pyhf.schema.validate(patchset, 'patchset.json')


@pytest.mark.parametrize(
    'patchset_file',
    [
        'patchset_bad_label_pattern.json',
        'patchset_bad_no_patch_name.json',
        'patchset_bad_empty_patches.json',
        'patchset_bad_no_patch_values.json',
        'patchset_bad_no_digests.json',
        'patchset_bad_no_description.json',
        'patchset_bad_no_labels.json',
        'patchset_bad_invalid_digests.json',
        'patchset_bad_hepdata_reference.json',
        'patchset_bad_no_version.json',
    ],
)
def test_patchset_fail(datadir, patchset_file):
    patchset = json.load(open(datadir.joinpath(patchset_file)))
    with pytest.raises(pyhf.exceptions.InvalidSpecification):
        pyhf.schema.validate(patchset, 'patchset.json')


def test_defs_always_cached(
    socket_disabled,  # noqa: F811
    isolate_modules,
):
    """
    Schema definitions should always be loaded from the local files and cached at first import.
    Otherwise pyhf will crash in contexts where the jsonschema.RefResolver cannot lookup the definition by the schema-id
    (e.g. a cluster node without network access).
    """
    modules_to_clear = [name for name in sys.modules if name.split('.')[0] == 'pyhf']
    for module_name in modules_to_clear:
        del sys.modules[module_name]
    pyhf = importlib.import_module('pyhf')

    spec = {
        'channels': [
            {
                'name': 'singlechannel',
                'samples': [
                    {
                        'name': 'signal',
                        'data': [10],
                        'modifiers': [
                            {'name': 'mu', 'type': 'normfactor', 'data': None}
                        ],
                    },
                    {
                        'name': 'background',
                        'data': [20],
                        'modifiers': [
                            {
                                'name': 'uncorr_bkguncrt',
                                'type': 'shapesys',
                                'data': [30],
                            }
                        ],
                    },
                ],
            }
        ]
    }
    pyhf.schema.validate(spec, 'model.json')  # may try to access network and fail


def test_schema_tensor_type_allowed(backend):
    tensorlib, _ = backend
    spec = {
        "channels": [
            {
                "name": "singlechannel",
                "samples": [
                    {
                        "name": "signal",
                        "data": tensorlib.astensor([10]),
                        "modifiers": [
                            {"name": "mu", "type": "normfactor", "data": None}
                        ],
                    },
                    {
                        "name": "background",
                        "data": tensorlib.astensor([15]),
                        "modifiers": [
                            {
                                "name": "uncorr_bkguncrt",
                                "type": "shapesys",
                                "data": tensorlib.astensor([5]),
                            }
                        ],
                    },
                ],
            }
        ]
    }
    assert pyhf.schema.validate(spec, "model.json") is None


def test_schema_tensor_type_disallowed(mocker, backend):
    tensorlib, _ = backend
    mocker.patch.object(
        pyhf.schema.validate,
        "__kwdefaults__",
        {"version": None, "allow_tensors": False},
    )
    spec = {
        "channels": [
            {
                "name": "singlechannel",
                "samples": [
                    {
                        "name": "signal",
                        "data": tensorlib.astensor([10]),
                        "modifiers": [
                            {"name": "mu", "type": "normfactor", "data": None}
                        ],
                    },
                    {
                        "name": "background",
                        "data": tensorlib.astensor([15]),
                        "modifiers": [
                            {
                                "name": "uncorr_bkguncrt",
                                "type": "shapesys",
                                "data": tensorlib.astensor([5]),
                            }
                        ],
                    },
                ],
            }
        ]
    }
    with pytest.raises(pyhf.exceptions.InvalidSpecification):
        pyhf.schema.validate(spec, "model.json")
