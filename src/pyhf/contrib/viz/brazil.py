"""Brazil Band Plots."""
import numpy as np
from matplotlib.container import Container


class BrazilBandContainer(Container):
    r"""
    Container for the :obj:`matplotlib.artist` objects created in a
    :func:`~pyhf.contrib.viz.brazil.plot_results` plot.

    The container can be treated like a :obj:`collections.namedtuple`
    ``(cls_obs, cls_exp, one_sigma_band, two_sigma_band, test_size)``.

    Attributes:
        cls_obs (:class:`matplotlib.lines.Line2D`): The artist of the
         :math:`\mathrm{CL}_{s,\mathrm{obs}}` line.

        cls_exp (:obj:`list` of :class:`matplotlib.lines.Line2D`): The artists of
         the :math:`\mathrm{CL}_{s,\mathrm{exp}}` lines.

        one_sigma_band (:class:`matplotlib.collections.PolyCollection`):
         The artists of the :math:`\mathrm{CL}_{s,\mathrm{exp}}` :math:`\pm1\sigma`
         band.

        two_sigma_band (:class:`matplotlib.collections.PolyCollection`):
         The artists of the :math:`\mathrm{CL}_{s,\mathrm{exp}}` :math:`\pm2\sigma`
         band.

        test_size (:class:`matplotlib.lines.Line2D`): The artist of the test size
         line.
    """

    def __init__(self, brazil_band_artists, **kwargs):
        r"""
        Args:
            brazil_band_artists (:obj:`tuple`): Tuple of
             ``(cls_obs, cls_exp, one_sigma_band, two_sigma_band, test_size)``.

              * ``cls_obs`` contains the :class:`matplotlib.lines.Line2D` of the
                observed :math:`\mathrm{CL}_{s}` line.

              * ``cls_exp`` is a :obj:`list` of :class:`matplotlib.lines.Line2D` of
                the expected :math:`\mathrm{CL}_{s}` lines.

              * ``one_sigma_band`` contains the :class:`matplotlib.collections.PolyCollection`
                of the :math:`\mathrm{CL}_{s,\mathrm{exp}}` :math:`\pm1\sigma` bands.

              * ``two_sigma_band`` contains the :class:`matplotlib.collections.PolyCollection`
                of the :math:`\mathrm{CL}_{s,\mathrm{exp}}` :math:`\pm2\sigma` bands.

              * ``test_size`` contains the :class:`matplotlib.lines.Line2D` of the test size.
        """
        (
            cls_obs,
            cls_exp,
            one_sigma_band,
            two_sigma_band,
            test_size,
        ) = brazil_band_artists
        self.cls_obs = cls_obs
        self.cls_exp = cls_exp
        self.one_sigma_band = one_sigma_band
        self.two_sigma_band = two_sigma_band
        self.test_size = test_size
        super().__init__(brazil_band_artists, **kwargs)


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
        >>> artists = pyhf.contrib.viz.brazil.plot_results(ax, poi_vals, results)

    Args:
        ax (:obj:`matplotlib.axes.Axes`): The matplotlib axis object to plot on.
        mutests (:obj:`list` or :obj:`array`): The values of the POI where the
          hypothesis tests were performed.
        tests (:obj:`list` or :obj:`array`): The :math:`\mathrm{CL}_{s}` values
          from the hypothesis tests.
        test_size (:obj:`float`): The size, :math:`\alpha`, of the test.

    Returns:
        :obj:`~pyhf.contrib.viz.brazil.BrazilBandContainer`: A container of :obj:`matplotlib.artist` drawn.
    """

    plot_components = kwargs.pop("components", False)
    no_cls = kwargs.pop("no_cls", False)

    if plot_components and len(tests[0]) != 3:
        raise ValueError(
            f"The components of 'tests' should have len of 3 to visualize the CLs components but have len {len(tests[0])}."
            + "\n'tests' should have format of: [CLs_obs, [CLsb, CLb], [CLs_exp band]]"
        )

    cls_obs = np.array([test[0] for test in tests]).flatten()
    if len(tests[0]) == 3:
        # split into components
        tail_probs = np.array([test[1] for test in tests])
        CLs_exp_set = np.array([test[2] for test in tests])
    else:
        CLs_exp_set = np.array([test[1] for test in tests])
    cls_exp = [
        np.array([exp_set[sigma_idx] for exp_set in CLs_exp_set]).flatten()
        for sigma_idx in range(5)
    ]

    if not no_cls:
        line_color = kwargs.pop("color", "black")
        (cls_obs_line,) = ax.plot(
            mutests, cls_obs, color=line_color, label=r"$\mathrm{CL}_{s}$"
        )

        cls_exp_lines = []
        for idx, color in zip(range(5), 5 * [line_color]):
            (_cls_exp_line,) = ax.plot(
                mutests,
                cls_exp[idx],
                color=color,
                linestyle="dotted" if idx != 2 else "dashed",
                label=None if idx != 2 else r"$\mathrm{CL}_{s,\mathrm{exp}}$",
            )
            cls_exp_lines.append(_cls_exp_line)
        one_sigma_band = ax.fill_between(
            mutests,
            cls_exp[0],
            cls_exp[-1],
            facecolor="yellow",
            label=r"$\pm2\sigma$ $\mathrm{CL}_{s,\mathrm{exp}}$",
        )
        two_sigma_band = ax.fill_between(
            mutests,
            cls_exp[1],
            cls_exp[-2],
            facecolor="green",
            label=r"$\pm1\sigma$ $\mathrm{CL}_{s,\mathrm{exp}}$",
        )

    test_size_color = kwargs.pop("test_size_color", "red")
    test_size_linestyle = kwargs.pop("test_size_linestyle", "solid")
    (test_size_line,) = ax.plot(
        mutests,
        [test_size] * len(mutests),
        color=test_size_color,
        linestyle=test_size_linestyle,
        label=rf"$\alpha={test_size}$",
    )
    ax.set_ylim(0, 1)

    x_label = kwargs.pop("xlabel", r"$\mu$ (POI)")
    y_label = kwargs.pop("ylabel", r"$\mathrm{CL}_{s}$")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if plot_components:
        _plot_cls_components(ax, mutests, tail_probs, **kwargs)

    # Order legend: ensure CLs expected band and test size are last in legend
    handles, labels = ax.get_legend_handles_labels()
    if not no_cls:
        for label_part in ["exp", "pm1", "pm2", "alpha"]:
            label_idx = [
                idx for idx, label in enumerate(labels) if label_part in label
            ][0]
            handles.append(handles.pop(label_idx))
            labels.append(labels.pop(label_idx))

    ax.legend(handles, labels, loc="best")

    return BrazilBandContainer(
        (
            cls_obs_line,
            cls_exp_lines,
            one_sigma_band,
            two_sigma_band,
            test_size_line,
        )
    )


def _plot_cls_components(ax, mutests, tail_probs, **kwargs):
    CLsb_obs = np.array([tail_prob[0] for tail_prob in tail_probs])
    CLb_obs = np.array([tail_prob[1] for tail_prob in tail_probs])

    axis_artists = []

    linewidth = kwargs.pop("linewidth", 2)
    no_clsb = kwargs.pop("no_clsb", False)
    no_clb = kwargs.pop("no_clb", False)
    if not no_clsb:
        CLsb_color = kwargs.pop("clsb_color", "red")
        CLsb_obs_line_artist = ax.plot(
            mutests,
            CLsb_obs,
            color=CLsb_color,
            linewidth=linewidth,
            label=r"$\mathrm{CL}_{s+b}$",
        )
        axis_artists.append(CLsb_obs_line_artist)
    if not no_clb:
        CLb_color = kwargs.pop("clb_color", "blue")
        CLb_obs_line_artist = ax.plot(
            mutests,
            CLb_obs,
            color=CLb_color,
            linewidth=linewidth,
            label=r"$\mathrm{CL}_{b}$",
        )
        axis_artists.append(CLb_obs_line_artist)


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
        >>> artists = pyhf.contrib.viz.brazil.plot_cls_components(ax, poi_vals, results)

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

    Returns:
        :obj:`list`: The list of lists and tuples of :obj:`matplotlib.artist` drawn.
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
    axis_artists = []
    if not no_cls:
        CLs_color = kwargs.pop("cls_color", "black")
        test_size_linestyle = kwargs.pop("test_size_linestyle", "dashdot")
        brazil_band_artists = plot_results(
            ax,
            mutests,
            CLs_results,
            test_size,
            ylabel=y_label,
            color=CLs_color,
            test_size_linestyle=test_size_linestyle,
        )
        axis_artists.append(brazil_band_artists)
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
        CLsb_obs_line_artist = ax.plot(
            mutests,
            CLsb_obs,
            color=CLsb_color,
            linewidth=linewidth,
            label=r"$\mathrm{CL}_{s+b}$",
        )
        axis_artists.append(CLsb_obs_line_artist)
    if not no_clb:
        CLb_color = kwargs.pop("clb_color", "blue")
        CLb_obs_line_artist = ax.plot(
            mutests,
            CLb_obs,
            color=CLb_color,
            linewidth=linewidth,
            label=r"$\mathrm{CL}_{b}$",
        )
        axis_artists.append(CLb_obs_line_artist)

    # Order legend: ensure CLs expected band and test size are last in legend
    handles, labels = ax.get_legend_handles_labels()
    if not no_cls:

        for label_part in ["exp", "pm1", "pm2", "alpha"]:
            label_idx = [
                idx for idx, label in enumerate(labels) if label_part in label
            ][0]
            handles.append(handles.pop(label_idx))
            labels.append(labels.pop(label_idx))

    ax.legend(handles, labels, loc="best")
    return axis_artists
