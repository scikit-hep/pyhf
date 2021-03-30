"""Brazil Band Plots."""
import numpy as np


def plot_results(ax, mutests, tests, test_size=0.05, **kwargs):
    r"""
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
        ax (:obj:`matplotlib.axes.Axes`): The matplotlib axis object to plot on.
        mutests (:obj:`list` or :obj:`array`): The values of the POI where the
          hypothesis tests were performed.
        tests (:obj:`list` or :obj:`array`): The :math:`\mathrm{CL}_{s}` values
          from the hypothesis tests.
        test_size (:obj:`float`): The size, :math:`\alpha`, of the test.
    """
    cls_obs = np.array([test[0] for test in tests]).flatten()
    cls_exp = [np.array([test[1][i] for test in tests]).flatten() for i in range(5)]

    line_color = kwargs.pop("color", "black")
    ax.plot(mutests, cls_obs, color=line_color, label=r"$\mathrm{CL}_{s}$")

    for idx, color in zip(range(5), 5 * [line_color]):
        ax.plot(
            mutests, cls_exp[idx], c=color, linestyle='dotted' if idx != 2 else 'dashed'
        )
    ax.fill_between(
        mutests,
        cls_exp[0],
        cls_exp[-1],
        facecolor="yellow",
        label=r"$\pm2\sigma$ $\mathrm{CL}_{s,\mathrm{exp}}$",
    )
    ax.fill_between(
        mutests,
        cls_exp[1],
        cls_exp[-2],
        facecolor="green",
        label=r"$\pm1\sigma$ $\mathrm{CL}_{s,\mathrm{exp}}$",
    )

    test_size_color = kwargs.pop("test_size_color", "red")
    ax.plot(
        mutests,
        [test_size] * len(mutests),
        color=test_size_color,
        label=rf"$\alpha={test_size}$",
    )
    ax.set_ylim(0, 1)

    x_label = kwargs.pop("xlabel", r"$\mu$ (POI)")
    y_label = kwargs.pop("ylabel", r"$\mathrm{CL}_{s}$")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)


def plot_cls_components(
    ax,
    mutests,
    tests,
    test_size=0.05,
    no_clb=False,
    no_clsb=False,
    no_cls=False,
    **kwargs,
):
    r"""
    Plot the values of :math:`\mathrm{CL}_{s+b}` and :math:`\mathrm{CL}_{b}`
    --- the components of the :math:`\mathrm{CL}_{s}` ratio --- on top of the
    :math:`\mathrm{CL}_{s}` values for a series of hypothesis tests for various
    POI values.

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
        ...     pyhf.infer.hypotest(
        ...         test_poi, data, model, return_expected_set=True, return_tail_probs=True
        ...     )
        ...     for test_poi in poi_vals
        ... ]
        >>> fig, ax = plt.subplots()
        >>> pyhf.contrib.viz.brazil.plot_cls_components(ax, poi_vals, results)

    Args:
        ax (:obj:`matplotlib.axes.Axes`): The matplotlib axis object to plot on.
        mutests (:obj:`list` or :obj:`array`): The values of the POI where the
          hypothesis tests were performed.
        tests (:obj:`list` or :obj:`array`): The collection of :math:`p`-values
          from the hypothesis tests.
          ``tests`` is required to have the same structure as
          :func:`pyhf.infer.hypotest`'s return when using ``return_expected_set=True``
          and ``return_tail_probs=True``: a tuple of :math:`\mathrm{CL}_{s}`,
          :math:`\left[\mathrm{CL}_{s+b}, \mathrm{CL}_{b}\right]`,
          :math:`\mathrm{CL}_{s,\mathrm{exp}}` band.
        test_size (:obj:`float`): The size, :math:`\alpha`, of the test.
        no_clb (:obj:`bool`): Bool for not plotting the :math:`\mathrm{CL}_{b}`
          component.
        no_clsb (:obj:`bool`): Bool for not plotting the :math:`\mathrm{CL}_{s+b}`
          component.
        no_cls (:obj:`bool`): Bool for not plotting the :math:`\mathrm{CL}_{s}`
          values.
    """

    if len(tests[0]) != 3:
        raise ValueError(
            f"The components of 'tests' should have len of 3 but have len {len(tests[0])}."
            + "\n'tests' should have format of: CLs_obs, [CLsb, CLb], [CLs_exp band]"
        )

    # split into components
    CLs_obs = np.array([test[0] for test in tests])
    tail_probs = np.array([test[1] for test in tests])
    CLs_exp_set = np.array([test[2] for test in tests])

    # zip CLs_obs and CLs_exp_set back into format for plot_results
    CLs_results = [(obs, exp_set) for obs, exp_set in zip(CLs_obs, CLs_exp_set)]

    # plot CLs_obs and CLs_expected set
    y_label = kwargs.pop("ylabel", r"$p\,$-value")
    if not no_cls:
        CLs_color = kwargs.pop("cls_color", "black")
        plot_results(
            ax, mutests, CLs_results, test_size, ylabel=y_label, color=CLs_color
        )
    else:
        # Still need to setup the canvas
        ax.set_ylim(0, 1)

        x_label = kwargs.pop("xlabel", r"$\mu$ (POI)")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

    CLsb_obs = np.array([tail_prob[0] for tail_prob in tail_probs])
    CLb_obs = np.array([tail_prob[1] for tail_prob in tail_probs])

    linewidth = kwargs.pop("linewidth", 2)
    if not no_clsb:
        CLsb_color = kwargs.pop("clsb_color", "red")
        ax.plot(
            mutests,
            CLsb_obs,
            color=CLsb_color,
            linewidth=linewidth,
            label=r"$\mathrm{CL}_{s+b}$",
        )
    if not no_clb:
        CLb_color = kwargs.pop("clb_color", "blue")
        ax.plot(
            mutests,
            CLb_obs,
            color=CLb_color,
            linewidth=linewidth,
            label=r"$\mathrm{CL}_{b}$",
        )

    # Place test size last in legend
    handles, labels = ax.get_legend_handles_labels()
    if not no_cls:
        test_size_idx = [idx for idx, label in enumerate(labels) if "alpha" in label][0]
        handles.append(handles.pop(test_size_idx))
        labels.append(labels.pop(test_size_idx))

    ax.legend(handles, labels, loc="best")
