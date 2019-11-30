from . import Model


def hepdata_like(signal_data, bkg_data, bkg_uncerts, batch_size=None):
    spec = {
        'channels': [
            {
                'name': 'singlechannel',
                'samples': [
                    {
                        'name': 'signal',
                        'data': signal_data,
                        'modifiers': [
                            {'name': 'mu', 'type': 'normfactor', 'data': None}
                        ],
                    },
                    {
                        'name': 'background',
                        'data': bkg_data,
                        'modifiers': [
                            {
                                'name': 'uncorr_bkguncrt',
                                'type': 'shapesys',
                                'data': bkg_uncerts,
                            }
                        ],
                    },
                ],
            }
        ]
    }
    return Model(spec, batch_size=batch_size)
