import pyhf
import pyhf.readxml
import pytest
import pyhf.exceptions
import json
import logging
import pyhf.workspace
import pyhf.utils
import copy


@pytest.fixture(
    scope='session',
    params=[
        (
            'validation/xmlimport_input/config/example.xml',
            'validation/xmlimport_input/',
        ),
        (
            'validation/xmlimport_input2/config/example.xml',
            'validation/xmlimport_input2',
        ),
        (
            'validation/xmlimport_input3/config/examples/example_ShapeSys.xml',
            'validation/xmlimport_input3',
        ),
    ],
    ids=['example-one', 'example-two', 'example-three'],
)
def workspace_xml(request):
    return pyhf.readxml.parse(*request.param)


@pytest.fixture(scope='function')
def workspace_factory(workspace_xml):
    return lambda: pyhf.Workspace(workspace_xml)


def test_build_workspace(workspace_factory):
    w = workspace_factory()
    assert w


def test_version_workspace(workspace_factory):
    ws = workspace_factory()
    assert ws.version is not None


def test_build_model(workspace_factory):
    w = workspace_factory()
    assert w.model()


def test_get_measurement_default(workspace_factory):
    w = workspace_factory()
    m = w.get_measurement()
    assert m


def test_get_measurement(workspace_factory):
    w = workspace_factory()
    for measurement in w.measurement_names:
        m = w.get_measurement(measurement_name=measurement)
        assert m['name'] == measurement
    for measurement_idx in range(len(w.measurement_names)):
        m = w.get_measurement(measurement_index=measurement_idx)
        assert m['name'] == w.measurement_names[measurement_idx]


def test_get_measurement_fake(workspace_factory):
    w = workspace_factory()
    m = w.get_measurement(poi_name='fake_poi')
    assert m


def test_get_measurement_nonexist(workspace_factory):
    w = workspace_factory()
    with pytest.raises(pyhf.exceptions.InvalidMeasurement) as excinfo:
        w.get_measurement(measurement_name='nonexistent_measurement')
    assert 'nonexistent_measurement' in str(excinfo.value)


def test_get_measurement_index_outofbounds(workspace_factory):
    ws = workspace_factory()
    with pytest.raises(pyhf.exceptions.InvalidMeasurement) as excinfo:
        ws.get_measurement(measurement_index=9999)
    assert 'out of bounds' in str(excinfo.value)


def test_get_measurement_no_measurements_defined(workspace_factory):
    ws = workspace_factory()
    ws.measurement_names = []
    with pytest.raises(pyhf.exceptions.InvalidMeasurement) as excinfo:
        ws.get_measurement()
    assert 'No measurements have been defined' in str(excinfo.value)


def test_get_workspace_measurement_priority(workspace_factory):
    w = workspace_factory()

    # does poi_name override all others?
    m = w.get_measurement(
        poi_name='fake_poi', measurement_name='FakeMeasurement', measurement_index=999
    )
    assert m['config']['poi'] == 'fake_poi'

    # does measurement_name override measurement_index?
    m = w.get_measurement(
        measurement_name=w.measurement_names[0], measurement_index=999
    )
    assert m['name'] == w.measurement_names[0]
    # only in cases where we have more than one measurement to pick from
    if len(w.measurement_names) > 1:
        assert m['name'] != w.measurement_names[-1]


def test_get_measurement_schema_validation(mocker, workspace_factory):
    mocker.patch('pyhf.utils.validate', return_value=None)
    assert pyhf.utils.validate.called is False
    w = workspace_factory()
    assert pyhf.utils.validate.call_count == 1
    assert pyhf.utils.validate.call_args[0][1] == 'workspace.json'
    w.get_measurement()
    assert pyhf.utils.validate.call_count == 2
    assert pyhf.utils.validate.call_args[0][1] == 'measurement.json'


def test_get_workspace_repr(workspace_factory):
    w = workspace_factory()
    assert 'pyhf.workspace.Workspace' in str(w)


def test_get_workspace_model_default(workspace_factory):
    w = workspace_factory()
    m = w.model()
    assert m


def test_workspace_observations(workspace_factory):
    w = workspace_factory()
    assert w.observations


@pytest.mark.parametrize(
    "with_aux",
    [True, False],
)
def test_get_workspace_data(workspace_factory, with_aux):
    w = workspace_factory()
    m = w.model()
    assert w.data(m, with_aux=with_aux)


def test_get_workspace_data_bad_model(workspace_factory, caplog):
    w = workspace_factory()
    m = w.model()
    # the iconic fragrance of an expected failure
    m.config.channels = [c.replace('channel', 'chanel') for c in m.config.channels]
    with caplog.at_level(logging.INFO, 'pyhf.pdf'):
        with pytest.raises(KeyError):
            assert w.data(m)
            assert 'Invalid channel' in caplog.text


def test_json_serializable(workspace_factory):
    assert json.dumps(workspace_factory())


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(channels=['fake-name']),
        dict(samples=['fake-sample']),
        dict(modifiers=['fake-modifier']),
        dict(modifier_types=['fake-type']),
    ],
)
def test_prune_error(workspace_factory, kwargs):
    ws = workspace_factory()
    with pytest.raises(pyhf.exceptions.InvalidWorkspaceOperation):
        ws.prune(**kwargs)


def test_prune_channel(workspace_factory):
    ws = workspace_factory()
    channel = ws.channels[0]
    with pytest.raises(pyhf.exceptions.InvalidWorkspaceOperation):
        ws.prune(channels=channel)

    if len(ws.channels) == 1:
        with pytest.raises(pyhf.exceptions.InvalidSpecification):
            ws.prune(channels=[channel])
    else:
        new_ws = ws.prune(channels=[channel])
        assert channel not in new_ws.channels
        assert channel not in [obs['name'] for obs in new_ws['observations']]


def test_prune_sample(workspace_factory):
    ws = workspace_factory()
    sample = ws.samples[1]
    with pytest.raises(pyhf.exceptions.InvalidWorkspaceOperation):
        ws.prune(samples=sample)

    new_ws = ws.prune(samples=[sample])
    assert sample not in new_ws.samples


def test_prune_modifier(workspace_factory):
    ws = workspace_factory()
    modifier = 'lumi'
    with pytest.raises(pyhf.exceptions.InvalidWorkspaceOperation):
        ws.prune(modifiers=modifier)

    new_ws = ws.prune(modifiers=[modifier])
    assert modifier not in new_ws.model().config.parameters
    assert modifier not in [
        p['name']
        for measurement in new_ws['measurements']
        for p in measurement['config']['parameters']
    ]


def test_prune_modifier_type(workspace_factory):
    ws = workspace_factory()
    modifier_type = 'lumi'

    with pytest.raises(pyhf.exceptions.InvalidWorkspaceOperation):
        ws.prune(modifier_types=modifier_type)

    new_ws = ws.prune(modifier_types=[modifier_type])
    assert modifier_type not in [item[1] for item in new_ws.modifiers]


def test_prune_measurements(workspace_factory):
    ws = workspace_factory()
    measurement = ws.measurement_names[0]

    if len(ws.measurement_names) == 1:
        with pytest.raises(pyhf.exceptions.InvalidWorkspaceOperation):
            ws.prune(measurements=measurement)
        with pytest.raises(pyhf.exceptions.InvalidSpecification):
            ws.prune(measurements=[measurement])
    else:
        new_ws = ws.prune(measurements=[measurement])
        assert new_ws
        assert measurement not in new_ws.measurement_names

        new_ws_list = ws.prune(measurements=[measurement])
        assert new_ws_list == new_ws


def test_rename_channel(workspace_factory):
    ws = workspace_factory()
    channel = ws.channels[0]
    renamed = 'renamedChannel'
    assert renamed not in ws.channels
    new_ws = ws.rename(channels={channel: renamed})
    assert channel not in new_ws.channels
    assert renamed in new_ws.channels
    assert channel not in [obs['name'] for obs in new_ws['observations']]
    assert renamed in [obs['name'] for obs in new_ws['observations']]


def test_rename_sample(workspace_factory):
    ws = workspace_factory()
    sample = ws.samples[1]
    renamed = 'renamedSample'
    assert renamed not in ws.samples
    new_ws = ws.rename(samples={sample: renamed})
    assert sample not in new_ws.samples
    assert renamed in new_ws.samples


def test_rename_modifier(workspace_factory):
    ws = workspace_factory()
    modifier = ws.model().config.parameters[0]
    renamed = 'renamedModifier'
    assert renamed not in ws.model().config.parameters
    new_ws = ws.rename(modifiers={modifier: renamed})
    assert modifier not in new_ws.model().config.parameters
    assert renamed in new_ws.model().config.parameters


def test_rename_poi(workspace_factory):
    ws = workspace_factory()
    poi = ws.get_measurement()['config']['poi']
    renamed = 'renamedPoi'
    assert renamed not in ws.model().config.parameters
    new_ws = ws.rename(modifiers={poi: renamed})
    assert poi not in new_ws.model().config.parameters
    assert renamed in new_ws.model().config.parameters
    assert new_ws.get_measurement()['config']['poi'] == renamed


def test_rename_measurement(workspace_factory):
    ws = workspace_factory()
    measurement = ws.measurement_names[0]
    renamed = 'renamedMeasurement'
    assert renamed not in ws.measurement_names
    new_ws = ws.rename(measurements={measurement: renamed})
    assert measurement not in new_ws.measurement_names
    assert renamed in new_ws.measurement_names


@pytest.fixture(scope='session')
def join_items():
    left = [
        {'name': 'left', 'key': 'value', 'deep': [{'name': 1}]},
        {'name': 'common', 'key': 'left', 'deep': [{'name': 1}]},
    ]
    right = [
        {'name': 'right', 'key': 'value', 'deep': [{'name': 2}]},
        {'name': 'common', 'key': 'right', 'deep': [{'name': 2}]},
    ]
    return (left, right)


def test_join_items_none(join_items):
    left_items, right_items = join_items
    joined = pyhf.workspace._join_items('none', left_items, right_items, key='name')
    assert all(left in joined for left in left_items)
    assert all(right in joined for right in right_items)


def test_join_items_outer(join_items):
    left_items, right_items = join_items
    joined = pyhf.workspace._join_items('outer', left_items, right_items, key='name')
    assert all(left in joined for left in left_items)
    assert all(right in joined for right in right_items)


def test_join_items_left_outer(join_items):
    left_items, right_items = join_items
    joined = pyhf.workspace._join_items(
        'left outer', left_items, right_items, key='name'
    )
    assert all(left in joined for left in left_items)
    assert not all(right in joined for right in right_items)


def test_join_items_right_outer(join_items):
    left_items, right_items = join_items
    joined = pyhf.workspace._join_items(
        'right outer', left_items, right_items, key='name'
    )
    assert not all(left in joined for left in left_items)
    assert all(right in joined for right in right_items)


def test_join_items_outer_deep(join_items):
    left_items, right_items = join_items
    joined = pyhf.workspace._join_items(
        'outer', left_items, right_items, key='name', deep_merge_key='deep'
    )
    assert [k['deep'] for k in joined if k['name'] == 'common'][0] == [
        {'name': 1},
        {'name': 2},
    ]


def test_join_items_left_outer_deep(join_items):
    left_items, right_items = join_items
    joined = pyhf.workspace._join_items(
        'left outer', left_items, right_items, key='name', deep_merge_key='deep'
    )
    assert [k['deep'] for k in joined if k['name'] == 'common'][0] == [
        {'name': 1},
        {'name': 2},
    ]


def test_join_items_right_outer_deep(join_items):
    left_items, right_items = join_items
    joined = pyhf.workspace._join_items(
        'right outer', left_items, right_items, key='name', deep_merge_key='deep'
    )
    assert [k['deep'] for k in joined if k['name'] == 'common'][0] == [
        {'name': 2},
        {'name': 1},
    ]


@pytest.mark.parametrize("join", ['none', 'outer'])
def test_combine_workspace_same_channels_incompatible_structure(
    workspace_factory, join
):
    ws = workspace_factory()
    new_ws = ws.rename(
        samples={ws.samples[0]: 'sample_other'},
    )
    with pytest.raises(pyhf.exceptions.InvalidWorkspaceOperation) as excinfo:
        pyhf.Workspace.combine(ws, new_ws, join=join)
    assert 'channel1' in str(excinfo.value)


@pytest.mark.parametrize("join", ['outer', 'left outer', 'right outer'])
def test_combine_workspace_same_channels_outer_join(workspace_factory, join):
    ws = workspace_factory()
    new_ws = ws.rename(channels={ws.channels[-1]: 'new_channel'})
    combined = pyhf.Workspace.combine(ws, new_ws, join=join)
    assert all(channel in combined.channels for channel in ws.channels)
    assert all(channel in combined.channels for channel in new_ws.channels)


@pytest.mark.parametrize("join", ['left outer', 'right outer'])
def test_combine_workspace_same_channels_outer_join_unsafe(
    workspace_factory, join, caplog
):
    ws = workspace_factory()
    new_ws = ws.rename(channels={ws.channels[-1]: 'new_channel'})
    pyhf.Workspace.combine(ws, new_ws, join=join)
    assert 'using an unsafe join operation' in caplog.text


@pytest.mark.parametrize("join", ['none', 'outer'])
def test_combine_workspace_incompatible_poi(workspace_factory, join):
    ws = workspace_factory()
    new_ws = ws.rename(
        channels={channel: f'renamed_{channel}' for channel in ws.channels},
        modifiers={ws.get_measurement()['config']['poi']: 'renamedPOI'},
    )
    with pytest.raises(pyhf.exceptions.InvalidWorkspaceOperation) as excinfo:
        pyhf.Workspace.combine(ws, new_ws, join=join)
    assert 'GaussExample' in str(excinfo.value)


@pytest.mark.parametrize("join", ['none', 'outer', 'left outer', 'right outer'])
def test_combine_workspace_diff_version(workspace_factory, join):
    ws = workspace_factory()
    ws.version = '1.0.0'
    new_ws = ws.rename(
        channels={channel: f'renamed_{channel}' for channel in ws.channels},
        samples={sample: f'renamed_{sample}' for sample in ws.samples},
        modifiers={
            modifier: f'renamed_{modifier}'
            for modifier, _ in ws.modifiers
            if not modifier == 'lumi'
        },
        measurements={
            measurement: f'renamed_{measurement}'
            for measurement in ws.measurement_names
        },
    )
    new_ws['version'] = '1.2.0'
    with pytest.raises(pyhf.exceptions.InvalidWorkspaceOperation) as excinfo:
        pyhf.Workspace.combine(ws, new_ws, join=join)
    assert '1.0.0' in str(excinfo.value)
    assert '1.2.0' in str(excinfo.value)


@pytest.mark.parametrize("join", ['none'])
def test_combine_workspace_duplicate_parameter_configs(workspace_factory, join):
    ws = workspace_factory()
    new_ws = ws.rename(
        channels={channel: f'renamed_{channel}' for channel in ws.channels},
    )
    with pytest.raises(pyhf.exceptions.InvalidWorkspaceOperation) as excinfo:
        pyhf.Workspace.combine(ws, new_ws, join=join)
    assert 'GaussExample' in str(excinfo.value)


@pytest.mark.parametrize("join", ['outer', 'left outer', 'right outer'])
def test_combine_workspace_duplicate_parameter_configs_outer_join(
    workspace_factory, join
):
    ws = workspace_factory()
    new_ws = ws.rename(
        channels={channel: f'renamed_{channel}' for channel in ws.channels},
    )
    combined = pyhf.Workspace.combine(ws, new_ws, join=join)

    poi = ws.get_measurement(measurement_name='GaussExample')['config']['poi']
    ws_parameter_configs = [
        parameter['name']
        for parameter in ws.get_measurement(measurement_name='GaussExample')['config'][
            'parameters'
        ]
    ]
    new_ws_parameter_configs = [
        parameter['name']
        for parameter in new_ws.get_measurement(measurement_name='GaussExample')[
            'config'
        ]['parameters']
    ]
    combined_parameter_configs = [
        parameter['name']
        for parameter in combined.get_measurement(measurement_name='GaussExample')[
            'config'
        ]['parameters']
    ]

    assert poi in ws_parameter_configs
    assert poi in new_ws_parameter_configs
    assert poi in combined_parameter_configs
    assert 'lumi' in ws_parameter_configs
    assert 'lumi' in new_ws_parameter_configs
    assert 'lumi' in combined_parameter_configs
    assert len(combined_parameter_configs) == len(set(combined_parameter_configs))


def test_combine_workspace_parameter_configs_ordering(workspace_factory):
    ws = workspace_factory()
    new_ws = ws.rename(
        channels={channel: f'renamed_{channel}' for channel in ws.channels},
    )
    assert (
        ws.get_measurement(measurement_name='GaussExample')['config']['parameters']
        == new_ws.get_measurement(measurement_name='GaussExample')['config'][
            'parameters'
        ]
    )


def test_combine_workspace_observation_ordering(workspace_factory):
    ws = workspace_factory()
    new_ws = ws.rename(
        channels={channel: f'renamed_{channel}' for channel in ws.channels},
    )
    assert ws['observations'][0]['data'] == new_ws['observations'][0]['data']


def test_combine_workspace_deepcopied(workspace_factory):
    ws = workspace_factory()
    new_ws = ws.rename(
        channels={channel: f'renamed_{channel}' for channel in ws.channels},
    )
    new_ws.get_measurement(measurement_name='GaussExample')['config']['parameters'][0][
        'bounds'
    ] = [[0.0, 1.0]]
    new_ws['observations'][0]['data'][0] = -10.0
    assert (
        ws.get_measurement(measurement_name='GaussExample')['config']['parameters'][0][
            'bounds'
        ]
        != new_ws.get_measurement(measurement_name='GaussExample')['config'][
            'parameters'
        ][0]['bounds']
    )
    assert ws['observations'][0]['data'] != new_ws['observations'][0]['data']


@pytest.mark.parametrize("join", ['fake join operation'])
def test_combine_workspace_invalid_join_operation(workspace_factory, join):
    ws = workspace_factory()
    new_ws = ws.rename(
        channels={channel: f'renamed_{channel}' for channel in ws.channels},
    )
    with pytest.raises(ValueError) as excinfo:
        pyhf.Workspace.combine(ws, new_ws, join=join)
    assert join in str(excinfo.value)


@pytest.mark.parametrize("join", ['none'])
def test_combine_workspace_invalid_join_operation_merge(workspace_factory, join):
    ws = workspace_factory()
    new_ws = ws.rename(
        channels={channel: f'renamed_{channel}' for channel in ws.channels},
    )
    with pytest.raises(ValueError) as excinfo:
        pyhf.Workspace.combine(ws, new_ws, join=join, merge_channels=True)
    assert join in str(excinfo.value)


@pytest.mark.parametrize("join", ['none'])
def test_combine_workspace_incompatible_parameter_configs(workspace_factory, join):
    ws = workspace_factory()
    new_ws = ws.rename(
        channels={channel: f'renamed_{channel}' for channel in ws.channels},
    )
    new_ws.get_measurement(measurement_name='GaussExample')['config']['parameters'][0][
        'bounds'
    ] = [[0.0, 1.0]]
    with pytest.raises(pyhf.exceptions.InvalidWorkspaceOperation) as excinfo:
        pyhf.Workspace.combine(ws, new_ws, join=join)
    assert 'GaussExample' in str(excinfo.value)


@pytest.mark.parametrize("join", ['outer'])
def test_combine_workspace_incompatible_parameter_configs_outer_join(
    workspace_factory, join
):
    ws = workspace_factory()
    new_ws = ws.rename(
        channels={channel: f'renamed_{channel}' for channel in ws.channels},
    )
    new_ws.get_measurement(measurement_name='GaussExample')['config']['parameters'][0][
        'bounds'
    ] = [[0.0, 1.0]]
    with pytest.raises(pyhf.exceptions.InvalidWorkspaceOperation) as excinfo:
        pyhf.Workspace.combine(ws, new_ws, join=join)
    assert 'GaussExample' in str(excinfo.value)
    assert ws.get_measurement(measurement_name='GaussExample')['config']['parameters'][
        0
    ]['name'] in str(excinfo.value)
    assert new_ws.get_measurement(measurement_name='GaussExample')['config'][
        'parameters'
    ][0]['name'] in str(excinfo.value)


@pytest.mark.parametrize("join", ['outer'])
def test_combine_workspace_compatible_parameter_configs_outer_join(
    workspace_factory, join
):
    ws = workspace_factory()
    left_parameters = ws.get_measurement(measurement_name='GaussExample')['config'][
        'parameters'
    ]
    right_parameters = ws.get_measurement(measurement_name='GaussExample')['config'][
        'parameters'
    ]
    assert pyhf.workspace._join_parameter_configs(
        'GaussExample', left_parameters, right_parameters
    )
    assert pyhf.workspace._join_measurements(
        join, ws['measurements'], ws['measurements']
    )


@pytest.mark.parametrize("join", ['outer'])
def test_combine_workspace_measurements_outer_join(workspace_factory, join):
    ws = workspace_factory()
    left_measurements = ws['measurements']
    right_measurements = copy.deepcopy(ws['measurements'])
    right_measurements[0]['config']['parameters'][0]['name'] = 'fake'
    assert pyhf.workspace._join_measurements(
        join, left_measurements, right_measurements
    )


def test_combine_workspace_incompatible_parameter_configs_left_outer_join(
    workspace_factory,
):
    ws = workspace_factory()
    new_ws = ws.rename(
        channels={channel: f'renamed_{channel}' for channel in ws.channels},
    )
    new_ws.get_measurement(measurement_name='GaussExample')['config']['parameters'][0][
        'bounds'
    ] = [[0.0, 1.0]]
    combined = pyhf.Workspace.combine(ws, new_ws, join='left outer')
    assert (
        combined.get_measurement(measurement_name='GaussExample')['config'][
            'parameters'
        ][0]
        == ws.get_measurement(measurement_name='GaussExample')['config']['parameters'][
            0
        ]
    )


def test_combine_workspace_incompatible_parameter_configs_right_outer_join(
    workspace_factory,
):
    ws = workspace_factory()
    new_ws = ws.rename(
        channels={channel: f'renamed_{channel}' for channel in ws.channels},
    )
    new_ws.get_measurement(measurement_name='GaussExample')['config']['parameters'][0][
        'bounds'
    ] = [[0.0, 1.0]]
    combined = pyhf.Workspace.combine(ws, new_ws, join='right outer')
    assert (
        combined.get_measurement(measurement_name='GaussExample')['config'][
            'parameters'
        ][0]
        == new_ws.get_measurement(measurement_name='GaussExample')['config'][
            'parameters'
        ][0]
    )


@pytest.mark.parametrize("join", ['none', 'outer'])
def test_combine_workspace_incompatible_observations(workspace_factory, join):
    ws = workspace_factory()
    new_ws = ws.rename(
        channels={channel: f'renamed_{channel}' for channel in ws.channels},
        samples={sample: f'renamed_{sample}' for sample in ws.samples},
        modifiers={
            modifier: f'renamed_{modifier}'
            for modifier, _ in ws.modifiers
            if not modifier == 'lumi'
        },
        measurements={
            measurement: f'renamed_{measurement}'
            for measurement in ws.measurement_names
        },
    )
    new_ws['observations'][0]['name'] = ws['observations'][0]['name']
    new_ws['observations'][0]['data'][0] = -10.0
    with pytest.raises(pyhf.exceptions.InvalidWorkspaceOperation) as excinfo:
        pyhf.Workspace.combine(ws, new_ws, join=join)
    assert ws['observations'][0]['name'] in str(excinfo.value)
    assert 'observations' in str(excinfo.value)


def test_combine_workspace_incompatible_observations_left_outer(workspace_factory):
    ws = workspace_factory()
    new_ws = ws.rename(
        channels={channel: f'renamed_{channel}' for channel in ws.channels},
        samples={sample: f'renamed_{sample}' for sample in ws.samples},
        modifiers={
            modifier: f'renamed_{modifier}'
            for modifier, _ in ws.modifiers
            if not modifier == 'lumi'
        },
        measurements={
            measurement: f'renamed_{measurement}'
            for measurement in ws.measurement_names
        },
    )
    new_ws['observations'][0]['name'] = ws['observations'][0]['name']
    new_ws['observations'][0]['data'][0] = -10.0
    combined = pyhf.Workspace.combine(ws, new_ws, join='left outer')
    assert (
        combined.observations[ws['observations'][0]['name']]
        == ws['observations'][0]['data']
    )


def test_combine_workspace_incompatible_observations_right_outer(workspace_factory):
    ws = workspace_factory()
    new_ws = ws.rename(
        channels={channel: f'renamed_{channel}' for channel in ws.channels},
        samples={sample: f'renamed_{sample}' for sample in ws.samples},
        modifiers={
            modifier: f'renamed_{modifier}'
            for modifier, _ in ws.modifiers
            if not modifier == 'lumi'
        },
        measurements={
            measurement: f'renamed_{measurement}'
            for measurement in ws.measurement_names
        },
    )
    new_ws['observations'][0]['name'] = ws['observations'][0]['name']
    new_ws['observations'][0]['data'][0] = -10.0
    combined = pyhf.Workspace.combine(ws, new_ws, join='right outer')
    assert (
        combined.observations[ws['observations'][0]['name']]
        == new_ws['observations'][0]['data']
    )


@pytest.mark.parametrize("join", pyhf.Workspace.valid_joins)
def test_combine_workspace(workspace_factory, join):
    ws = workspace_factory()
    new_ws = ws.rename(
        channels={channel: f'renamed_{channel}' for channel in ws.channels},
        samples={sample: f'renamed_{sample}' for sample in ws.samples},
        modifiers={
            modifier: f'renamed_{modifier}'
            for modifier, _ in ws.modifiers
            if not modifier == 'lumi'
        },
        measurements={
            measurement: f'renamed_{measurement}'
            for measurement in ws.measurement_names
        },
    )
    combined = pyhf.Workspace.combine(ws, new_ws, join=join)
    assert set(combined.channels) == set(ws.channels + new_ws.channels)
    assert set(combined.samples) == set(ws.samples + new_ws.samples)
    assert set(combined.model().config.parameters) == set(
        ws.model().config.parameters + new_ws.model().config.parameters
    )


def test_workspace_equality(workspace_factory):
    ws = workspace_factory()
    ws_other = workspace_factory()
    assert ws == ws
    assert ws == ws_other
    assert ws != 'not a workspace'


def test_workspace_inheritance(workspace_factory):
    ws = workspace_factory()
    new_ws = ws.rename(
        channels={channel: f'renamed_{channel}' for channel in ws.channels},
        samples={sample: f'renamed_{sample}' for sample in ws.samples},
        modifiers={
            modifier: f'renamed_{modifier}'
            for modifier, _ in ws.modifiers
            if not modifier == 'lumi'
        },
        measurements={
            measurement: f'renamed_{measurement}'
            for measurement in ws.measurement_names
        },
    )

    class FooWorkspace(pyhf.Workspace):
        pass

    combined = FooWorkspace.combine(ws, new_ws)
    assert isinstance(combined, FooWorkspace)


@pytest.mark.parametrize("join", ['outer', 'left outer', 'right outer'])
def test_combine_workspace_merge_channels(workspace_factory, join):
    ws = workspace_factory()
    new_ws = ws.prune(samples=ws.samples[1:]).rename(
        samples={ws.samples[0]: f'renamed_{ws.samples[0]}'}
    )
    combined_ws = pyhf.Workspace.combine(ws, new_ws, join=join, merge_channels=True)
    assert new_ws.samples[0] in combined_ws.samples
    assert any(
        sample['name'] == new_ws.samples[0]
        for sample in combined_ws['channels'][0]['samples']
    )


def test_sorted(workspace_factory):
    ws = workspace_factory()
    # force the first sample in each channel to be last
    for channel in ws['channels']:
        channel['samples'][0]['name'] = 'zzzzlast'

    new_ws = pyhf.Workspace.sorted(ws)
    for channel in ws['channels']:
        # check no sort
        assert channel['samples'][0]['name'] == 'zzzzlast'
    for channel in new_ws['channels']:
        # check sort
        assert channel['samples'][-1]['name'] == 'zzzzlast'


def test_closure_over_workspace_build():
    model = pyhf.simplemodels.hepdata_like(
        signal_data=[12.0, 11.0], bkg_data=[50.0, 52.0], bkg_uncerts=[3.0, 7.0]
    )
    data = [51, 48]
    one = pyhf.infer.hypotest(1.0, data + model.config.auxdata, model)

    workspace = pyhf.Workspace.build(model, data)

    assert json.dumps(workspace)

    newmodel = workspace.model()
    newdata = workspace.data(newmodel)
    two = pyhf.infer.hypotest(1.0, newdata, newmodel)

    assert one == two

    newworkspace = pyhf.Workspace.build(newmodel, newdata)

    assert pyhf.utils.digest(newworkspace) == pyhf.utils.digest(workspace)


def test_wspace_immutable():
    model = pyhf.simplemodels.hepdata_like(
        signal_data=[12.0, 11.0], bkg_data=[50.0, 52.0], bkg_uncerts=[3.0, 7.0]
    )
    data = [51, 48]
    workspace = pyhf.Workspace.build(model, data)

    spec = json.loads(json.dumps(workspace))

    ws = pyhf.Workspace(spec)
    model = ws.model()
    before = model.config.suggested_init()
    spec["measurements"][0]["config"]["parameters"][0]["inits"] = [1.5]

    model = ws.model()
    after = model.config.suggested_init()

    assert before == after
