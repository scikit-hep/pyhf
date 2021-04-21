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
    results_plot_container = brazil.plot_results(
        ax, data["testmus"], data["results"], test_size=0.05
    )
    brazil_band_container = results_plot_container.brazil_band

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


def test_cls_components_container(datadir):
    data = json.load(open(datadir.join("tail_probs_hypotest_results.json")))

    fig = Figure()
    ax = fig.subplots()
    results_plot_container = brazil.plot_results(
        ax, data["testmus"], data["results"], test_size=0.05, components=True
    )
    cls_components_container = results_plot_container.cls_components

    assert len(cls_components_container) == 2
    assert cls_components_container == (
        cls_components_container.clsb,
        cls_components_container.clb,
    )

    assert cls_components_container.clsb is not None
    assert cls_components_container.clb is not None


def test_results_plot_container(datadir):
    data = json.load(open(datadir.join("tail_probs_hypotest_results.json")))

    fig = Figure()
    ax = fig.subplots()
    results_plot_container = brazil.plot_results(
        ax, data["testmus"], data["results"], test_size=0.05, components=True
    )

    assert len(results_plot_container) == 2
    assert results_plot_container == (
        results_plot_container.brazil_band,
        results_plot_container.cls_components,
    )

    assert results_plot_container.brazil_band is not None
    assert results_plot_container.cls_components is not None


@pytest.mark.mpl_image_compare
def test_plot_results(datadir):
    data = json.load(open(datadir.join("hypotest_results.json")))

    fig = Figure()
    ax = fig.subplots()
    brazil.plot_results(ax, data["testmus"], data["results"], test_size=0.05)

    return fig


@pytest.mark.mpl_image_compare
def test_plot_results_components(datadir):
    data = json.load(open(datadir.join("tail_probs_hypotest_results.json")))

    fig = Figure()
    ax = fig.subplots()
    brazil.plot_results(
        ax, data["testmus"], data["results"], test_size=0.05, components=True
    )
    return fig


@pytest.mark.mpl_image_compare
def test_plot_results_components_no_clb(datadir):
    data = json.load(open(datadir.join("tail_probs_hypotest_results.json")))

    fig = Figure()
    ax = fig.subplots()
    results_plot_container = brazil.plot_results(
        ax,
        data["testmus"],
        data["results"],
        test_size=0.05,
        components=True,
        no_clb=True,
    )
    cls_components_container = results_plot_container.cls_components

    assert cls_components_container.clsb is not None
    assert cls_components_container.clb is None
    return fig


@pytest.mark.mpl_image_compare
def test_plot_results_components_no_clsb(datadir):
    data = json.load(open(datadir.join("tail_probs_hypotest_results.json")))

    fig = Figure()
    ax = fig.subplots()
    results_plot_container = brazil.plot_results(
        ax,
        data["testmus"],
        data["results"],
        test_size=0.05,
        components=True,
        no_clsb=True,
    )
    cls_components_container = results_plot_container.cls_components

    assert cls_components_container.clsb is None
    assert cls_components_container.clb is not None
    return fig


@pytest.mark.mpl_image_compare
def test_plot_results_components_no_cls(datadir):
    data = json.load(open(datadir.join("tail_probs_hypotest_results.json")))

    fig = Figure()
    ax = fig.subplots()
    results_plot_container = brazil.plot_results(
        ax,
        data["testmus"],
        data["results"],
        test_size=0.05,
        components=True,
        no_cls=True,
    )
    assert results_plot_container.brazil_band is None
    assert results_plot_container.cls_components is not None
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
            ax, data["testmus"], data["results"], test_size=0.05, components=True
        )
