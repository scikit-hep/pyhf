import pyhf
import pyhf.readxml
import pytest
import pyhf.exceptions
import json

@pytest.mark.parametrize(
    'toplvl, basedir',
    [
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
def test_build_workspace(toplvl, basedir):
    assert pyhf.Workspace(pyhf.readxml.parse(toplvl, basedir))
