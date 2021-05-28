import warnings

import pyhf


def test_correlated_background():
    model = pyhf.simplemodels.correlated_background(
        signal=[12.0, 11.0],
        bkg=[50.0, 52.0],
        bkg_up=[45.0, 57.0],
        bkg_down=[55.0, 47.0],
    )
    assert model.config.channels == ["single_channel"]
    assert model.config.samples == ["background", "signal"]
    assert model.config.par_order == ["mu", "correlated_bkg_uncertainty"]
    assert model.config.suggested_init() == [1.0, 0.0]


def test_uncorrelated_background():
    model = pyhf.simplemodels.uncorrelated_background(
        signal=[12.0, 11.0], bkg=[50.0, 52.0], bkg_uncertainty=[3.0, 7.0]
    )
    assert model.config.channels == ["singlechannel"]
    assert model.config.samples == ["background", "signal"]
    assert model.config.par_order == ["mu", "uncorr_bkguncrt"]
    assert model.config.suggested_init() == [1.0, 1.0, 1.0]


# TODO: Remove after pyhf v0.6.2 is released
def test_deprecated_apis():
    with warnings.catch_warnings(record=True) as _warning:
        # Cause all warnings to always be triggered
        warnings.simplefilter("always")
        pyhf.simplemodels.hepdata_like([12.0, 11.0], [50.0, 52.0], [3.0, 7.0])
        assert len(_warning) == 1
        assert issubclass(_warning[-1].category, DeprecationWarning)
        assert (
            "pyhf.simplemodels.hepdata_like is deprecated in favor of pyhf.simplemodels.uncorrelated_background"
            in str(_warning[-1].message)
        )
