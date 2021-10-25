from warnings import warn

from pyhf import Model

__all__ = ["correlated_background", "uncorrelated_background"]


def __dir__():
    return __all__


def correlated_background(signal, bkg, bkg_up, bkg_down, batch_size=None):
    r"""
    Construct a simple single channel :class:`~pyhf.pdf.Model` with a
    :class:`~pyhf.modifiers.histosys` modifier representing a background
    with a fully correlated bin-by-bin uncertainty.

    Args:
        signal (:obj:`list`): The data in the signal sample.
        bkg (:obj:`list`): The data in the background sample.
        bkg_up (:obj:`list`): The background sample under an upward variation
         corresponding to :math:`\alpha=+1`.
        bkg_down (:obj:`list`): The background sample under a downward variation
         corresponding to :math:`\alpha=-1`.
        batch_size (:obj:`None` or :obj:`int`): Number of simultaneous (batched) Models to compute.

    Returns:
        ~pyhf.pdf.Model: The statistical model adhering to the :obj:`model.json` schema.

    Example:
        >>> import pyhf
        >>> pyhf.set_backend("numpy")
        >>> model = pyhf.simplemodels.correlated_background(
        ...     signal=[12.0, 11.0],
        ...     bkg=[50.0, 52.0],
        ...     bkg_up=[45.0, 57.0],
        ...     bkg_down=[55.0, 47.0],
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


def uncorrelated_background(signal, bkg, bkg_uncertainty, batch_size=None):
    """
    Construct a simple single channel :class:`~pyhf.pdf.Model` with a
    :class:`~pyhf.modifiers.shapesys` modifier representing an uncorrelated
    background uncertainty.

    Example:
        >>> import pyhf
        >>> pyhf.set_backend("numpy")
        >>> model = pyhf.simplemodels.uncorrelated_background(
        ...     signal=[12.0, 11.0], bkg=[50.0, 52.0], bkg_uncertainty=[3.0, 7.0]
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
        signal (:obj:`list`): The data in the signal sample
        bkg (:obj:`list`): The data in the background sample
        bkg_uncertainty (:obj:`list`): The statistical uncertainty on the background sample counts
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
                        'data': signal,
                        'modifiers': [
                            {'name': 'mu', 'type': 'normfactor', 'data': None}
                        ],
                    },
                    {
                        'name': 'background',
                        'data': bkg,
                        'modifiers': [
                            {
                                'name': 'uncorr_bkguncrt',
                                'type': 'shapesys',
                                'data': bkg_uncertainty,
                            }
                        ],
                    },
                ],
            }
        ]
    }
    return Model(spec, batch_size=batch_size)


# Deprecated APIs
def _deprecated_api_warning(
    deprecated_api, new_api, deprecated_release, remove_release
):
    warn(
        f"{deprecated_api} is deprecated in favor of {new_api} as of pyhf v{deprecated_release} and will be removed in release {remove_release}."
        + f" Please use {new_api}.",
        DeprecationWarning,
        stacklevel=3,  # Raise to user level
    )
