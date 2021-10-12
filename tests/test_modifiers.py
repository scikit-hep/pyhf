import pyhf

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
