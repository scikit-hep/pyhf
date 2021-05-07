import pyhf


def test_correlated_background():
    model = pyhf.simplemodels.correlated_background(
        signal=[12.0, 11.0],
        bkg=[50.0, 52.0],
        bkg_up=[53.0, 59.0],
        bkg_down=[45.0, 49.0],
    )
    assert model.config.channels == ["single_channel"]
    assert model.config.samples == ["background", "signal"]
    assert model.config.par_order == ["mu", "correlated_bkg_uncertainty"]
    assert model.config.suggested_init() == [1.0, 0.0]


def test_uncorrelated_background():
    model = pyhf.simplemodels.hepdata_like(
        signal_data=[12.0, 11.0], bkg_data=[50.0, 52.0], bkg_uncerts=[3.0, 7.0]
    )
    assert model.config.channels == ["singlechannel"]
    assert model.config.samples == ["background", "signal"]
    assert model.config.par_order == ["mu", "uncorr_bkguncrt"]
    assert model.config.suggested_init() == [1.0, 1.0, 1.0]
