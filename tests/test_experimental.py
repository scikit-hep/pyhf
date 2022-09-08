import pyhf
import pyhf.experimental.modifiers


def test_add_custom_modifier(backend):
    tensorlib, _ = backend

    new_params = {
        'm1': {'inits': (1.0,), 'bounds': ((-5.0, 5.0),)},
        'm2': {'inits': (1.0,), 'bounds': ((-5.0, 5.0),)},
    }

    expanded_pyhf = pyhf.experimental.modifiers.add_custom_modifier(
        'customfunc', ['m1', 'm2'], new_params
    )
    model = pyhf.Model(
        {
            'channels': [
                {
                    'name': 'singlechannel',
                    'samples': [
                        {
                            'name': 'signal',
                            'data': [10] * 20,
                            'modifiers': [
                                {
                                    'name': 'f2',
                                    'type': 'customfunc',
                                    'data': {'expr': 'm1'},
                                },
                            ],
                        },
                        {
                            'name': 'background',
                            'data': [100] * 20,
                            'modifiers': [
                                {
                                    'name': 'f1',
                                    'type': 'customfunc',
                                    'data': {'expr': 'm1+(m2**2)'},
                                },
                            ],
                        },
                    ],
                }
            ]
        },
        modifier_set=expanded_pyhf,
        poi_name='m1',
        validate=False,
        batch_size=1,
    )

    assert tensorlib.tolist(model.expected_actualdata([[1.0, 2.0]])) == [
        [
            510.0,
            510.0,
            510.0,
            510.0,
            510.0,
            510.0,
            510.0,
            510.0,
            510.0,
            510.0,
            510.0,
            510.0,
            510.0,
            510.0,
            510.0,
            510.0,
            510.0,
            510.0,
            510.0,
            510.0,
        ]
    ]
