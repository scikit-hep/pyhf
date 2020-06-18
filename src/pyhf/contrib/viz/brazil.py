"""Brazil Band Plots."""
import numpy as np


def plot_results(ax, mutests, tests, test_size=0.05):
    """Plot a series of hypothesis tests for various POI values."""
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
