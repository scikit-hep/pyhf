"""Brazil Band Plots."""
import numpy as np


def plot_results(ax, mutests, tests, test_size=0.05):
    """
    Plot a series of hypothesis tests for various POI values.

    Example:

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> import pyhf
        >>> import pyhf.contrib.viz.brazil
        >>> pyhf.set_backend("numpy")
        >>> model = pyhf.simplemodels.hepdata_like(
        ...     signal_data=[12.0, 11.0], bkg_data=[50.0, 52.0], bkg_uncerts=[3.0, 7.0]
        ... )
        >>> observations = [51, 48]
        >>> data = observations + model.config.auxdata
        >>> poi_vals = np.linspace(0, 5, 41)
        >>> results = [
        ...     pyhf.infer.hypotest(test_poi, data, model, return_expected_set=True)
        ...     for test_poi in poi_vals
        ... ]
        >>> fig, ax = plt.subplots()
        >>> pyhf.contrib.viz.brazil.plot_results(ax, poi_vals, results)

    Args:
        ax (`matplotlib.axes.Axes`): The matplotlib axis object to plot on.
        mutests (:obj:`list` or :obj:`array`): The values of the POI where the
          hypothesis tests were performed.
        tests (:obj:`list` or :obj:`array`): The :math:`\\mathrm{CL}_{s}` values
          from the hypothesis tests.
        test_size (:obj:`float`): The size, :math:`\\alpha`, of the test.
    """
    cls_obs = np.array([test[0] for test in tests]).flatten()
    cls_exp = [np.array([test[1][i] for test in tests]).flatten() for i in range(5)]
    ax.plot(mutests, cls_obs, c='black')
    for idx, color in zip(range(5), 5 * ['black']):
        ax.plot(
            mutests, cls_exp[idx], c=color, linestyle='dotted' if idx != 2 else 'dashed'
        )
    ax.fill_between(mutests, cls_exp[0], cls_exp[-1], facecolor='yellow')
    ax.fill_between(mutests, cls_exp[1], cls_exp[-2], facecolor='green')
    ax.plot(mutests, [test_size] * len(mutests), c='red')
    ax.set_ylim(0, 1)

    ax.set_xlabel(r"$\mu$ (POI)")
    ax.set_ylabel(r"$\mathrm{CL}_{s}$")
