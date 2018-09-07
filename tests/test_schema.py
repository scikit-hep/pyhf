import pyhf
import pytest

def test_no_samples():
    spec = {
        'channels': [
            {
                'name': 'channel',
                'samples': []
            },
        ]
    }
    with pytest.raises(pyhf.exceptions.InvalidSpecification):
        pyhf.Model(spec)

def test_sample_missing_data():
    spec = {
        'channels': [
            {
                'name': 'channel',
                'samples': [
                    {
                        'name': 'sample',
                        'data': [],
                        'modifiers': []
                    }
                ]
            },
        ]
    }
    with pytest.raises(pyhf.exceptions.InvalidSpecification):
        pyhf.Model(spec)

def test_sample_missing_name():
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
        pyhf.Model(spec)

def test_sample_missing_modifiers():
    spec = {
        'channels': [
            {
                'name': 'channel',
                'samples': [
                    {
                        'name': 'sample',
                        'data': [10.],
                        'modifiers': []
                    }
                ]
            },
        ]
    }
    pyhf.Model(spec)

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
        pyhf.Model(spec)


def test_empty_staterror():
    spec = {
        'channels': [
            {
                'name': 'channel',
                'samples': [
                    {
                        'name': 'sample',
                        'data': [10.],
                        'modifiers': [
                            {'name': 'staterror_channel', 'type': 'staterror', 'data': []}
                        ]
                    }
                ]
            },
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
                        'data': [10.],
                        'modifiers': [
                            {'name': 'sample_norm', 'type': 'shapesys','data': []}
                        ]
                    }
                ]
            },
        ]
    }
    with pytest.raises(pyhf.exceptions.InvalidSpecification):
        pyhf.Model(spec)
