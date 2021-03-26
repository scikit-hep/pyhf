import pytest
import json
import pyhf.contrib.viz.brazil as brazil
import matplotlib.pyplot as plt

# Compare against a reference image generated via:
# pytest --mpl-generate-path=tests/contrib/baseline tests/contrib/test_viz.py


@pytest.mark.mpl_image_compare
def test_plot_results():
    data = json.load(open('tests/contrib/hypotestresults.json'))
    fig, ax = plt.subplots(1, 1)
    brazil.plot_results(ax, data['testmus'], data['results'], test_size=0.05)
    return fig


@pytest.mark.mpl_image_compare
def test_plot_cls_components():
    data = json.load(open("tests/contrib/tail_probs_hypotest_results.json"))
    fig, ax = plt.subplots(1, 1)
    brazil.plot_cls_components(ax, data["testmus"], data["results"], test_size=0.05)
    return fig


def test_plot_cls_components_data_structure():
    """
    test results should have format of: CLs_obs, [CLsb, CLb], [CLs_exp band]
    """
    data = json.load(open("tests/contrib/hypotestresults.json"))
    fig, ax = plt.subplots(1, 1)
    with pytest.raises(ValueError):
        brazil.plot_cls_components(ax, data["testmus"], data["results"], test_size=0.05)
