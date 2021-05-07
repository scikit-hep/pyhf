from . import Model

__all__ = ["hepdata_like"]


def __dir__():
    return __all__


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
        signal_data (:obj:`list`): The data in the signal sample
        bkg_data (:obj:`list`): The data in the background sample
        bkg_uncerts (:obj:`list`): The statistical uncertainty on the background sample counts
        batch_size (:obj:`None` or :obj:`int`): Number of simultaneous (batched) Models to compute

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


def correlated_background(signal, bkg, bkg_up, bkg_down, batch_size=None):
    r"""
    Construct a simple single channel :class:`~pyhf.pdf.Model` with a
    :class:`~pyhf.modifiers.histosys` modifier representing a correlated
    background uncertainty.

    Args:
        signal (:obj:`list`): The data in the signal sample.
        bkg (:obj:`list`): The data in the background sample.
        bkg_up (:obj:`list`): The background sample under a :math:`+1\sigma` variation.
        bkg_down (:obj:`list`): The background sample under a :math:`-1\sigma` variation.
        batch_size (:obj:`None` or :obj:`int`): Number of simultaneous (batched) Models to compute.

    Returns:
        ~pyhf.pdf.Model: The statistical model adhering to the :obj:`model.json` schema.

    Example:
        >>> import pyhf
        >>> pyhf.set_backend("numpy")
        >>> model = pyhf.simplemodels.correlated_background(
        ...     signal=[12.0, 11.0],
        ...     bkg=[50.0, 52.0],
        ...     bkg_up=[53.0, 59.0],
        ...     bkg_down=[45.0, 49.0],
        ... )
        >>> model.schema
        'model.json'
        >>> model.config.channels
        ['single_channel']
        >>> model.config.samples
        ['background', 'signal']
        >>> model.config.parameters
        ['correlated_bkg_uncertainty', 'mu']
        >>> model.expected_data(model.config.suggested_init())
        array([62., 63.,  0.])

    """
    spec = {
        "channels": [
            {
                "name": "single_channel",
                "samples": [
                    {
                        "name": "signal",
                        "data": signal,
                        "modifiers": [
                            {"name": "mu", "type": "normfactor", "data": None}
                        ],
                    },
                    {
                        "name": "background",
                        "data": bkg,
                        "modifiers": [
                            {
                                "name": "correlated_bkg_uncertainty",
                                "type": "histosys",
                                "data": {"hi_data": bkg_up, "lo_data": bkg_down},
                            }
                        ],
                    },
                ],
            }
        ]
    }
    return Model(spec, batch_size=batch_size)
