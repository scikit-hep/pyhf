import pyhf
import pytest
import os
import json
import pkg_resources

def test_schema_access():
    assert os.isfile(pkg_resources.resource_filename('pyhf','data/spec.json'))

def test_schema_access():
    assert json.load(open(pkg_resources.resource_filename('pyhf','data/spec.json')))

def test_missing_sample_name():
    spec = {
        'channels': [
            {
                'name': 'channel',
                'samples': [
                    {
                        'data': [1],
                        'modifiers': []
                    },
                ]
            }
        ]
    }
    with pytest.raises(pyhf.exceptions.InvalidSpecification):
        pyhf.hfpdf(spec)

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
                            {'name': 'a_name', 'type': 'this_should_not_exist', 'data': [1]}
                        ]
                    },
                ]
            }
        ]
    }
    with pytest.raises(pyhf.exceptions.InvalidSpecification):
        pyhf.hfpdf(spec)
