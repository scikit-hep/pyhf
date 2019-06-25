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
