import pytest
import json
import pyhf.contrib.viz.brazil as brazil
import matplotlib.pyplot as plt


@pytest.mark.mpl_image_compare
def test_brazil():
    data = json.load(open('tests/contrib/hypotestresults.json'))
    fig, ax = plt.subplots(1, 1)
    brazil.plot_results(ax, data['testmus'], data['results'], test_size=0.05)
    return fig
