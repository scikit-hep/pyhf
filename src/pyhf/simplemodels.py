from . import Model


def hepdata_like(signal_data, bkg_data, bkg_uncerts, batch_size=None):
    """
    Construct a simple single channel :class:`~pyhf.pdf.Model` with a
    :class:`~pyhf.modifiers.shapesys` modifier representing an uncorrelated
    background uncertainty.

    Example:
        >>> import pyhf
        >>> pyhf.set_backend("numpy")
        >>> model = pyhf.simplemodels.hepdata_like(
        ...     signal_data=[12.0, 11.0], bkg_data=[50.0, 52.0], bkg_uncerts=[3.0, 7.0]
        ... )
        >>> model.schema
        'model.json'
        >>> model.config.channels
        ['singlechannel']
        >>> model.config.samples
        ['background', 'signal']
        >>> model.config.parameters
        ['mu', 'uncorr_bkguncrt']
        >>> model.expected_data(model.config.suggested_init())
        array([ 62.        ,  63.        , 277.77777778,  55.18367347])

    Args:
        signal_data (`list`): The data in the signal sample
        bkg_data (`list`): The data in the background sample
        bkg_uncerts (`list`): The statistical uncertainty on the background sample counts
        batch_size (`None` or `int`): Number of simultaneous (batched) Models to compute

    Returns:
        ~pyhf.pdf.Model: The statistical model adhering to the :obj:`model.json` schema

    """
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
