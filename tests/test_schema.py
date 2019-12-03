import pyhf
import pytest


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
    pyhf.Model(spec, poiname='mypoi')


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
        pyhf.Model(spec, poiname='mypoi')


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
        pyhf.Model(spec, poiname='mypoi')


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
    pyhf.Model(spec, poiname='mypoi')


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
        pyhf.Model(spec, poiname='mypoi')


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
        pyhf.Model(spec, poiname='mypoi')
