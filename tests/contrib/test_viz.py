import pytest
import json
import pyhf.contrib.viz.brazil as brazil
import matplotlib.pyplot as plt

# Tests with the @pytest.mark.mpl_image_compare decorator compare against
# reference images generated via:
# pytest --mpl-generate-path=tests/contrib/baseline tests/contrib/test_viz.py


@pytest.mark.mpl_image_compare
def test_plot_results():
    data = json.load(open("tests/contrib/hypotest_results.json"))
    fig, ax = plt.subplots(1, 1)
    brazil.plot_results(ax, data['testmus'], data['results'], test_size=0.05)
    return fig


@pytest.mark.mpl_image_compare
def test_plot_cls_components():
    data = json.load(open("tests/contrib/tail_probs_hypotest_results.json"))
    fig, ax = plt.subplots(1, 1)
    brazil.plot_cls_components(ax, data["testmus"], data["results"], test_size=0.05)
    return fig


@pytest.mark.mpl_image_compare
def test_plot_cls_components_no_clb():
    data = json.load(open("tests/contrib/tail_probs_hypotest_results.json"))
    fig, ax = plt.subplots(1, 1)
    brazil.plot_cls_components(
        ax, data["testmus"], data["results"], test_size=0.05, no_clb=True
    )
    return fig


@pytest.mark.mpl_image_compare
def test_plot_cls_components_no_clsb():
    data = json.load(open("tests/contrib/tail_probs_hypotest_results.json"))
    fig, ax = plt.subplots(1, 1)
    brazil.plot_cls_components(
        ax, data["testmus"], data["results"], test_size=0.05, no_clsb=True
    )
    return fig


@pytest.mark.mpl_image_compare
def test_plot_cls_components_no_cls():
    data = json.load(open("tests/contrib/tail_probs_hypotest_results.json"))
    fig, ax = plt.subplots(1, 1)
    brazil.plot_cls_components(
        ax, data["testmus"], data["results"], test_size=0.05, no_cls=True
    )
    return fig


def test_plot_cls_components_data_structure():
    """
    test results should have format of: CLs_obs, [CLsb, CLb], [CLs_exp band]
    """
    data = json.load(open("tests/contrib/hypotest_results.json"))
    fig, ax = plt.subplots(1, 1)
    with pytest.raises(ValueError):
        brazil.plot_cls_components(ax, data["testmus"], data["results"], test_size=0.05)
