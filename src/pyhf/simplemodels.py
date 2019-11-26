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
    return Model(
        spec,
        batch_size=batch_size,
        modifier_settings={
            'normsys': {'interpcode': 'code4'},
            'histosys': {'interpcode': 'code4p'},
        },
    )


def overall_bkguncert(
    signal_data, bkg_data, bkg_uncert_hi, bkg_uncert_lo, batch_size=None
):
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
                                'name': 'corr_bkguncrt',
                                'type': 'normsys',
                                'data': {'hi': bkg_uncert_hi, 'lo': bkg_uncert_lo},
                            }
                        ],
                    },
                ],
            }
        ]
    }
    return Model(
        spec,
        batch_size=batch_size,
        modifier_settings={
            'normsys': {'interpcode': 'code4'},
            'histosys': {'interpcode': 'code4p'},
        },
    )
