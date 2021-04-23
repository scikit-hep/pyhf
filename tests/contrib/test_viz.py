import json

import matplotlib
import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure

import pyhf.contrib.viz.brazil as brazil

# Tests with the @pytest.mark.mpl_image_compare decorator compare against
# reference images generated via:
# pytest --mpl-generate-path=tests/contrib/baseline tests/contrib/test_viz.py


def test_brazil_band_collection(datadir):
    data = json.load(open(datadir.join("hypotest_results.json")))

    fig = Figure()
    ax = fig.subplots()
    brazil_band_collection = brazil.plot_results(
        data["testmus"], data["results"], test_size=0.05, ax=ax
    )

    assert brazil_band_collection.cls_obs is not None
    assert brazil_band_collection.cls_exp is not None
    assert len(brazil_band_collection.cls_exp) == 5
    assert brazil_band_collection.one_sigma_band is not None
    assert brazil_band_collection.two_sigma_band is not None
    assert brazil_band_collection.test_size is not None
    assert brazil_band_collection.clsb is None
    assert brazil_band_collection.clb is None

    data = json.load(open(datadir.join("tail_probs_hypotest_results.json")))

    fig = Figure()
    ax = fig.subplots()
    brazil_band_collection = brazil.plot_results(
        data["testmus"], data["results"], test_size=0.05, ax=ax, components=True
    )

    assert brazil_band_collection.cls_obs is not None
    assert brazil_band_collection.cls_exp is not None
    assert len(brazil_band_collection.cls_exp) == 5
    assert brazil_band_collection.one_sigma_band is not None
    assert brazil_band_collection.two_sigma_band is not None
    assert brazil_band_collection.test_size is not None
    assert brazil_band_collection.clsb is not None
    assert brazil_band_collection.clb is not None

    assert brazil_band_collection.axes == ax

    _fig = Figure()
    _ax = _fig.subplots()
    (line,) = _ax.plot([1, 2], [3, 4])
    artists = [*iter(brazil_band_collection)]
    artists[0] = line
    bad_collection = brazil.BrazilBandCollection(*artists)
    assert bad_collection.axes is None


@pytest.mark.mpl_image_compare
def test_plot_results(datadir):
    data = json.load(open(datadir.join("hypotest_results.json")))

    fig = Figure()
    ax = fig.subplots()
    brazil_band_collection = brazil.plot_results(
        data["testmus"], data["results"], test_size=0.05, ax=ax
    )
    assert brazil_band_collection.axes == ax

    return fig


@pytest.mark.mpl_image_compare
def test_plot_results_no_axis(datadir):
    data = json.load(open(datadir.join("hypotest_results.json")))

    matplotlib.use("agg")  # Use non-gui backend
    fig, ax = plt.subplots()
    ax.set_yscale("log")  # Also test log y detection
    brazil.plot_results(data["testmus"], data["results"], test_size=0.05)

    return fig


@pytest.mark.mpl_image_compare
def test_plot_results_components(datadir):
    data = json.load(open(datadir.join("tail_probs_hypotest_results.json")))

    fig = Figure()
    ax = fig.subplots()
    brazil.plot_results(
        data["testmus"], data["results"], test_size=0.05, ax=ax, components=True
    )
    return fig


@pytest.mark.mpl_image_compare
def test_plot_results_components_no_clb(datadir):
    data = json.load(open(datadir.join("tail_probs_hypotest_results.json")))

    fig = Figure()
    ax = fig.subplots()
    brazil_band_collection = brazil.plot_results(
        data["testmus"],
        data["results"],
        test_size=0.05,
        ax=ax,
        components=True,
        no_clb=True,
    )

    assert brazil_band_collection.clsb is not None
    assert brazil_band_collection.clb is None
    return fig


@pytest.mark.mpl_image_compare
def test_plot_results_components_no_clsb(datadir):
    data = json.load(open(datadir.join("tail_probs_hypotest_results.json")))

    fig = Figure()
    ax = fig.subplots()
    brazil_band_collection = brazil.plot_results(
        data["testmus"],
        data["results"],
        test_size=0.05,
        ax=ax,
        components=True,
        no_clsb=True,
    )

    assert brazil_band_collection.clsb is None
    assert brazil_band_collection.clb is not None
    return fig


@pytest.mark.mpl_image_compare
def test_plot_results_components_no_cls(datadir):
    data = json.load(open(datadir.join("tail_probs_hypotest_results.json")))

    fig = Figure()
    ax = fig.subplots()
    brazil_band_collection = brazil.plot_results(
        data["testmus"],
        data["results"],
        test_size=0.05,
        ax=ax,
        components=True,
        no_cls=True,
    )

    assert brazil_band_collection.cls_obs is None
    assert brazil_band_collection.cls_exp is None
    assert brazil_band_collection.one_sigma_band is None
    assert brazil_band_collection.two_sigma_band is None
    assert brazil_band_collection.test_size is None
    assert brazil_band_collection.clsb is not None
    assert brazil_band_collection.clb is not None
    return fig


def test_plot_results_components_data_structure(datadir):
    """
    test results should have format of: [CLs_obs, [CLsb, CLb], [CLs_exp band]]
    """
    data = json.load(open(datadir.join("hypotest_results.json")))

    fig = Figure()
    ax = fig.subplots()
    with pytest.raises(ValueError):
        brazil.plot_results(
            data["testmus"], data["results"], test_size=0.05, ax=ax, components=True
        )
