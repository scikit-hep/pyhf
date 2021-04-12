import json

import pytest
from matplotlib.figure import Figure

import pyhf.contrib.viz.brazil as brazil

# Tests with the @pytest.mark.mpl_image_compare decorator compare against
# reference images generated via:
# pytest --mpl-generate-path=tests/contrib/baseline tests/contrib/test_viz.py


@pytest.mark.mpl_image_compare
def test_plot_results(datadir):
    data = json.load(open(datadir.join("hypotest_results.json")))

    fig = Figure()
    ax = fig.subplots()
    brazil.plot_results(ax, data['testmus'], data['results'], test_size=0.05)
    return fig


@pytest.mark.mpl_image_compare
def test_plot_cls_components(datadir):
    data = json.load(open(datadir.join("tail_probs_hypotest_results.json")))

    fig = Figure()
    ax = fig.subplots()
    brazil.plot_cls_components(ax, data["testmus"], data["results"], test_size=0.05)
    return fig


@pytest.mark.mpl_image_compare
def test_plot_cls_components_no_clb(datadir):
    data = json.load(open(datadir.join("tail_probs_hypotest_results.json")))

    fig = Figure()
    ax = fig.subplots()
    brazil.plot_cls_components(
        ax, data["testmus"], data["results"], test_size=0.05, no_clb=True
    )
    return fig


@pytest.mark.mpl_image_compare
def test_plot_cls_components_no_clsb(datadir):
    data = json.load(open(datadir.join("tail_probs_hypotest_results.json")))

    fig = Figure()
    ax = fig.subplots()
    brazil.plot_cls_components(
        ax, data["testmus"], data["results"], test_size=0.05, no_clsb=True
    )
    return fig


@pytest.mark.mpl_image_compare
def test_plot_cls_components_no_cls(datadir):
    data = json.load(open(datadir.join("tail_probs_hypotest_results.json")))

    fig = Figure()
    ax = fig.subplots()
    brazil.plot_cls_components(
        ax, data["testmus"], data["results"], test_size=0.05, no_cls=True
    )
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
