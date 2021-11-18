import pyhf
import numpy

modifiers_to_test = [
    "histosys",
    "normfactor",
    "normsys",
    "shapefactor",
    "shapesys",
    "staterror",
]
modifier_pdf_types = ["normal", None, "normal", None, "poisson", "normal"]


def test_shapefactor_build():
    spec = {
        'channels': [
            {
                'name': 'channel',
                'samples': [
                    {
                        'name': 'sample',
                        'data': [10.0] * 3,
                        'modifiers': [
                            {'name': 'mu', 'type': 'normfactor', 'data': None},
                        ],
                    },
                    {
                        'name': 'another_sample',
                        'data': [5.0] * 3,
                        'modifiers': [
                            {'name': 'freeshape', 'type': 'shapefactor', 'data': None}
                        ],
                    },
                ],
            }
        ],
    }

    model = pyhf.Model(spec)
    assert model


def test_staterror_holes():
    spec = {
        'channels': [
            {
                'name': 'channel1',
                'samples': [
                    {
                        'name': 'another_sample',
                        'data': [50, 0, 0, 70],
                        'modifiers': [
                            {'name': 'mu', 'type': 'normfactor', 'data': None},
                            {
                                'name': 'staterror_1',
                                'type': 'staterror',
                                'data': [5, 0, 5, 5],
                            },
                        ],
                    },
                ],
            },
            {
                'name': 'channel2',
                'samples': [
                    {
                        'name': 'another_sample',
                        'data': [50, 0, 10, 70],
                        'modifiers': [
                            {
                                'name': 'staterror_2',
                                'type': 'staterror',
                                'data': [5, 0, 5, 5],
                            }
                        ],
                    },
                ],
            },
        ],
    }

    model = pyhf.Model(spec, poi_name="")
    assert model.config.npars == 9
    _, factors = model._modifications(
        pyhf.tensorlib.astensor([2, 2.0, 1.0, 1.0, 3.0, 4.0, 1.0, 5.0, 6.0])
    )
    assert model.config.param_set("staterror_1").suggested_fixed == [
        False,
        True,
        True,
        False,
    ]
    assert all(
        [
            isinstance(fixed, bool)
            for fixed in model.config.param_set("staterror_1").suggested_fixed
        ]
    )
    assert model.config.param_set("staterror_2").suggested_fixed == [
        False,
        True,
        False,
        False,
    ]
    assert all(
        [
            isinstance(fixed, bool)
            for fixed in model.config.param_set("staterror_2").suggested_fixed
        ]
    )
    assert (factors[1][0, 0, 0, :] == [2.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 1.0]).all()
    assert (factors[1][1, 0, 0, :] == [1.0, 1.0, 1.0, 1.0, 4.0, 1.0, 5.0, 6.0]).all()

    data = model.expected_data(model.config.suggested_init())
    assert numpy.isfinite(model.logpdf(model.config.suggested_init(), data)).all()


def test_shapesys_holes():
    spec = {
        'channels': [
            {
                'name': 'channel1',
                'samples': [
                    {
                        'name': 'another_sample',
                        'data': [50, 60, 0, 70],
                        'modifiers': [
                            {'name': 'mu', 'type': 'normfactor', 'data': None},
                            {
                                'name': 'freeshape1',
                                'type': 'shapesys',
                                'data': [5, 0, 5, 5],
                            },
                        ],
                    },
                ],
            },
            {
                'name': 'channel2',
                'samples': [
                    {
                        'name': 'another_sample',
                        'data': [50, 60, 0, 70],
                        'modifiers': [
                            {
                                'name': 'freeshape2',
                                'type': 'shapesys',
                                'data': [5, 0, 5, 5],
                            }
                        ],
                    },
                ],
            },
        ],
    }

    model = pyhf.Model(spec, poi_name="mu")
    _, factors = model._modifications(
        pyhf.tensorlib.astensor([1.0, 2.0, 1.0, 1.0, 3.0, 4.0, 1.0, 1.0, 5.0])
    )
    assert (factors[1][0, 0, 0, :] == [2.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 1.0]).all()
    assert (factors[1][1, 0, 0, :] == [1.0, 1.0, 1.0, 1.0, 4.0, 1.0, 1.0, 5.0]).all()

    assert model.config.param_set("freeshape1").suggested_fixed == [
        False,
        True,
        True,
        False,
    ]
    assert model.config.param_set("freeshape2").suggested_fixed == [
        False,
        True,
        True,
        False,
    ]
