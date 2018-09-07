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

def test_sample_missing_all_modifiers():
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
    with pytest.raises(pyhf.exceptions.InvalidModel):
        pyhf.Model(spec)

def test_one_sample_missing_modifiers():
    spec = {
        'channels': [
            {
                'name': 'channel',
                'samples': [
                    {
                        'name': 'sample',
                        'data': [10.],
                        'modifiers': []
                    },
                    {
                        'name': 'another_sample',
                        'data': [5.],
                        'modifiers': [{'name': 'mypoi', 'type': 'normfactor', 'data': None}]
                    }
                ]
            },
        ]
    }
    pyhf.Model(spec, poiname='mypoi')


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

def test_empty_histosys():
    spec = {
        'channels': [
            {
                'name': 'channel',
                'samples': [
                    {
                        'name': 'sample',
                        'data': [10.],
                        'modifiers': [
                            {'name': 'modifier', 'type': 'histosys', 'data': {'lo_data': [], 'hi_data': []}}
                        ]
                    }
                ]
            },
        ]
    }
    with pytest.raises(pyhf.exceptions.InvalidSpecification):
        pyhf.Model(spec)
