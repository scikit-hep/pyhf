import pyhf
import pyhf.readxml
import pytest
import pyhf.exceptions
import json


@pytest.fixture(
    scope='function',
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
def workspace_factory(request):
    return lambda: pyhf.Workspace(pyhf.readxml.parse(*request.param))


def test_build_workspace(workspace_factory):
    w = workspace_factory()
    assert w


def test_build_model(workspace_factory):
    w = workspace_factory()
    assert w.model()


def test_get_measurement(workspace_factory):
    w = workspace_factory()
    m = w.get_measurement()
    assert m


def test_get_measurement_fake(workspace_factory):
    w = workspace_factory()
    m = w.get_measurement(poi_name='fake_poi')
    assert m


"""
test the following
- check for workspace.json schema validation failure
- check that the schema validation is being called
- check that setting a measurement works (via checks to get_measurement())
"""
