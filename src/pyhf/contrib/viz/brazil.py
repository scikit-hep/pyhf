import numpy as np


def plot_results(ax, mutests, tests, test_size=0.05):
    cls_obs = np.array([test[0] for test in tests]).flatten()
    cls_exp = [np.array([test[1][i] for test in tests]).flatten() for i in range(5)]
    ax.plot(mutests, cls_obs, c='black')
    for i, c in zip(range(5), ['k', 'k', 'k', 'k', 'k']):
        ax.plot(mutests, cls_exp[i], c=c, linestyle='dotted' if i != 2 else 'dashed')
    ax.fill_between(mutests, cls_exp[0], cls_exp[-1], facecolor='yellow')
    ax.fill_between(mutests, cls_exp[1], cls_exp[-2], facecolor='g')
    ax.plot(mutests, [test_size] * len(mutests), c='r')
    ax.set_ylim(0, 1)
