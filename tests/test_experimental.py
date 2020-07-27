import pyhf.experimental.query
import pytest


@pytest.mark.parametrize(
    'expression,expected',
    [
        ('mu', [0]),
        ('mu[0]', 0),
        ('mu*', [0]),
        ('stat_*', [12, 13, 2, 3, 4, 5, 6]),
        ('stat_*[1]', [13, 3]),
        ('*_firstchannel', [2, 3, 4, 5, 6]),
        ('*channel', [12, 13, 2, 3, 4, 5, 6, 14, 15]),
        ('syst', [7, 8, 9, 10, 11]),
    ],
)
def test_parameter_indices(expression, expected):
    # NB: "asecondchannel" is used to test the sorting of indices as well
    spec = {
        'channels': [
            {
                'name': 'firstchannel',
                'samples': [
                    {
                        'name': 'mu',
                        'data': [10.0] * 5,
                        'modifiers': [
                            {'name': 'mu', 'type': 'normfactor', 'data': None},
                            {
                                'type': 'normsys',
                                'name': 'shape',
                                'data': {"hi": 0.5, "lo": 1.5},
                            },
                        ],
                    },
                    {
                        'name': 'bkg1',
                        'data': [50.0] * 5,
                        'modifiers': [
                            {
                                'name': 'stat_firstchannel',
                                'type': 'staterror',
                                'data': [12.0] * 5,
                            },
                            {"data": [10] * 5, "name": "syst", "type": "shapesys",},
                        ],
                    },
                    {
                        'name': 'bkg2',
                        'data': [30.0] * 5,
                        'modifiers': [
                            {
                                'name': 'stat_firstchannel',
                                'type': 'staterror',
                                'data': [5.0] * 5,
                            },
                            {
                                "data": [0] * 5,
                                "name": "syst_lowstats",
                                "type": "shapesys",
                            },
                        ],
                    },
                    {
                        'name': 'abkg3',
                        'data': [20.0] * 5,
                        'modifiers': [
                            {
                                'name': 'stat_firstchannel',
                                'type': 'staterror',
                                'data': [5.0] * 5,
                            },
                        ],
                    },
                ],
            },
            {
                'name': 'asecondchannel',
                'samples': [
                    {
                        'name': 'bkg1',
                        'data': [10.0] * 2,
                        'modifiers': [
                            {
                                'name': 'stat_asecondchannel',
                                'type': 'staterror',
                                'data': [4.0] * 2,
                            },
                            {
                                "data": [10] * 2,
                                "name": "syst_asecondchannel",
                                "type": "shapesys",
                            },
                        ],
                    },
                    {
                        'name': 'bkg2',
                        'data': [15.0] * 2,
                        'modifiers': [
                            {
                                'name': 'stat_asecondchannel',
                                'type': 'staterror',
                                'data': [5.0] * 2,
                            },
                            {
                                "data": [0] * 2,
                                "name": "syst_lowstats_asecondchannel",
                                "type": "shapesys",
                            },
                        ],
                    },
                ],
            },
        ]
    }
    pdf = pyhf.Model(spec)
    assert pyhf.experimental.query.parameter_indices(pdf, expression) == expected
