import json

import pytest
from matplotlib.figure import Figure

import pyhf.contrib.viz.brazil as brazil

# Tests with the @pytest.mark.mpl_image_compare decorator compare against
# reference images generated via:
# pytest --mpl-generate-path=tests/contrib/baseline tests/contrib/test_viz.py


def test_brazil_band_container(datadir):
    data = json.load(open(datadir.join("hypotest_results.json")))

    fig = Figure()
    ax = fig.subplots()
    brazil_band_container = brazil.plot_results(
        ax, data["testmus"], data["results"], test_size=0.05
    )

    assert len(brazil_band_container) == 5
    assert brazil_band_container == (
        brazil_band_container.cls_obs,
        brazil_band_container.cls_exp,
        brazil_band_container.one_sigma_band,
        brazil_band_container.two_sigma_band,
        brazil_band_container.test_size,
    )

    assert brazil_band_container.cls_obs is not None
    assert len(brazil_band_container.cls_exp) == 5
    assert brazil_band_container.one_sigma_band is not None
    assert brazil_band_container.two_sigma_band is not None
    assert brazil_band_container.test_size is not None
    assert brazil_band_container.axes is not None
    assert brazil_band_container.axes == brazil_band_container.cls_obs.axes


@pytest.mark.mpl_image_compare
def test_plot_results(datadir):
    data = json.load(open(datadir.join("hypotest_results.json")))

    fig = Figure()
    ax = fig.subplots()
    brazil.plot_results(ax, data["testmus"], data["results"], test_size=0.05)

    return fig


@pytest.mark.mpl_image_compare
def test_plot_cls_components(datadir):
    data = json.load(open(datadir.join("tail_probs_hypotest_results.json")))

    fig = Figure()
    ax = fig.subplots()
    brazil_band_container, clsb_artists, clb_artists = brazil.plot_cls_components(
        ax, data["testmus"], data["results"], test_size=0.05
    )
    assert len(brazil_band_container) == 5
    assert len(clsb_artists) == 1
    assert len(clb_artists) == 1
    return fig


@pytest.mark.mpl_image_compare
def test_plot_cls_components_no_clb(datadir):
    data = json.load(open(datadir.join("tail_probs_hypotest_results.json")))

    fig = Figure()
    ax = fig.subplots()
    brazil_band_container, clsb_artists = brazil.plot_cls_components(
        ax, data["testmus"], data["results"], test_size=0.05, no_clb=True
    )
    assert len(brazil_band_container) == 5
    assert len(clsb_artists) == 1
    return fig


@pytest.mark.mpl_image_compare
def test_plot_cls_components_no_clsb(datadir):
    data = json.load(open(datadir.join("tail_probs_hypotest_results.json")))

    fig = Figure()
    ax = fig.subplots()
    brazil_band_container, clb_artists = brazil.plot_cls_components(
        ax, data["testmus"], data["results"], test_size=0.05, no_clsb=True
    )
    assert len(brazil_band_container) == 5
    assert len(clb_artists) == 1
    return fig


@pytest.mark.mpl_image_compare
def test_plot_cls_components_no_cls(datadir):
    data = json.load(open(datadir.join("tail_probs_hypotest_results.json")))

    fig = Figure()
    ax = fig.subplots()
    clsb_artists, clb_artists = brazil.plot_cls_components(
        ax, data["testmus"], data["results"], test_size=0.05, no_cls=True
    )
    assert len(clsb_artists) == 1
    assert len(clb_artists) == 1
    return fig


def test_plot_cls_components_data_structure(datadir):
    """
    test results should have format of: CLs_obs, [CLsb, CLb], [CLs_exp band]
    """
    data = json.load(open(datadir.join("hypotest_results.json")))

    fig = Figure()
    ax = fig.subplots()
    with pytest.raises(ValueError):
        brazil.plot_cls_components(ax, data["testmus"], data["results"], test_size=0.05)
