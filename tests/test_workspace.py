import pyhf
import pyhf.readxml
import pytest
import pyhf.exceptions
import json
import logging


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
    with pytest.raises(pyhf.exceptions.InvalidMeasurement):
        m = w.get_measurement(measurement_name='nonexistent_measurement')


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
    m = w.get_measurement()
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


def test_get_workspace_data(workspace_factory):
    w = workspace_factory()
    m = w.model()
    assert w.data(m)


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


def test_prune_nothing(workspace_factory):
    ws = workspace_factory()
    new_ws = ws.prune(
        channels=['fake-name'],
        samples=['fake-sample'],
        modifiers=['fake-modifier'],
        modifier_types=['fake-type'],
    )


def test_prune_channel(workspace_factory):
    ws = workspace_factory()
    channel = ws.channels[0]
    if len(ws.channels) == 1:
        with pytest.raises(pyhf.exceptions.InvalidSpecification):
            new_ws = ws.prune(channels=[channel])
    else:
        new_ws = ws.prune(channels=channel)
        assert channel not in new_ws.channels
        assert channel not in [obs['name'] for obs in new_ws['observations']]


def test_prune_sample(workspace_factory):
    ws = workspace_factory()
    sample = ws.samples[1]
    new_ws = ws.prune(samples=[sample])
    assert new_ws
    assert sample not in new_ws.samples


def test_prune_modifier(workspace_factory):
    ws = workspace_factory()
    modifier = 'lumi'
    new_ws = ws.prune(modifiers=[modifier])
    assert new_ws
    assert modifier not in new_ws.parameters
    assert modifier not in [
        p['name']
        for measurement in new_ws['measurements']
        for p in measurement['config']['parameters']
    ]


def test_prune_modifier_type(workspace_factory):
    ws = workspace_factory()
    modifier_type = 'lumi'
    new_ws = ws.prune(modifier_types=[modifier_type])
    assert new_ws
    assert modifier_type not in [item[1] for item in new_ws.modifiers]


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
    modifier = ws.parameters[0]
    renamed = 'renamedModifier'
    assert renamed not in ws.parameters
    new_ws = ws.rename(modifiers={modifier: renamed})
    assert modifier not in new_ws.parameters
    assert renamed in new_ws.parameters


def test_combine_workspace(workspace_factory):
    ws = workspace_factory()
    new_ws = ws.rename(
        samples={
            'background1': 'background3',
            'background2': 'background4',
            'signal': 'signal2',
        },
        modifiers={
            'syst1': 'syst4',
            'bkg1Shape': 'bkg3Shape',
            'bkg2Shape': 'bkg4Shape',
        },
    )
    combined_combine = ws.combine(new_ws)
    assert set(combined_combine.channels) == set(ws.channels + new_ws.channels)
    assert set(combined_combine.samples) == set(ws.samples + new_ws.samples)
    assert set(combined_combine.parameters) == set(ws.parameters + new_ws.parameters)
    # check adding
    combined_add = ws + new_ws
    assert combined_add == combined_combine
    assert set(combined_add.channels) == set(ws.channels + new_ws.channels)
    assert set(combined_add.samples) == set(ws.samples + new_ws.samples)
    assert set(combined_add.parameters) == set(ws.parameters + new_ws.parameters)
