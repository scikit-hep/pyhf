import warnings

import pytest
import pyhf


@pytest.fixture(scope='function')
def default_backend(backend):
    pyhf.set_backend(*backend, default=True)
    yield backend


def test_correlated_background(backend):
    model = pyhf.simplemodels.correlated_background(
        signal=[12.0, 11.0],
        bkg=[50.0, 52.0],
        bkg_up=[45.0, 57.0],
        bkg_down=[55.0, 47.0],
    )
    assert model.config.channels == ["single_channel"]
    assert model.config.samples == ["background", "signal"]
    assert model.config.par_order == ["correlated_bkg_uncertainty", "mu"]
    assert model.config.par_names() == ['correlated_bkg_uncertainty', "mu"]
    assert model.config.suggested_init() == [0.0, 1.0]


def test_uncorrelated_background(backend):
    model = pyhf.simplemodels.uncorrelated_background(
        signal=[12.0, 11.0], bkg=[50.0, 52.0], bkg_uncertainty=[3.0, 7.0]
    )
    assert model.config.channels == ["singlechannel"]
    assert model.config.samples == ["background", "signal"]
    assert model.config.par_order == ["mu", "uncorr_bkguncrt"]
    assert model.config.par_names() == [
        'mu',
        'uncorr_bkguncrt[0]',
        'uncorr_bkguncrt[1]',
    ]
    assert model.config.suggested_init() == [1.0, 1.0, 1.0]


# See https://github.com/scikit-hep/pyhf/issues/1654
@pytest.mark.fail_pytorch
@pytest.mark.fail_pytorch64
@pytest.mark.fail_tensorflow
@pytest.mark.fail_jax
def test_correlated_background_default_backend(default_backend):
    model = pyhf.simplemodels.correlated_background(
        signal=[12.0, 11.0],
        bkg=[50.0, 52.0],
        bkg_up=[45.0, 57.0],
        bkg_down=[55.0, 47.0],
    )
    assert model.config.channels == ["single_channel"]
    assert model.config.samples == ["background", "signal"]
    assert model.config.par_order == ["correlated_bkg_uncertainty", "mu"]
    assert model.config.par_names() == ['correlated_bkg_uncertainty', "mu"]
    assert model.config.suggested_init() == [0.0, 1.0]


# See #1654
@pytest.mark.fail_pytorch
@pytest.mark.fail_pytorch64
@pytest.mark.fail_tensorflow
@pytest.mark.fail_jax
def test_uncorrelated_background_default_backend(default_backend):
    model = pyhf.simplemodels.uncorrelated_background(
        signal=[12.0, 11.0], bkg=[50.0, 52.0], bkg_uncertainty=[3.0, 7.0]
    )
    assert model.config.channels == ["singlechannel"]
    assert model.config.samples == ["background", "signal"]
    assert model.config.par_order == ["mu", "uncorr_bkguncrt"]
    assert model.config.par_names() == [
        'mu',
        'uncorr_bkguncrt[0]',
        'uncorr_bkguncrt[1]',
    ]
    assert model.config.suggested_init() == [1.0, 1.0, 1.0]


# TODO: Remove when pyhf.simplemodels.hepdata_like is removed in pyhf v0.7.0
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
