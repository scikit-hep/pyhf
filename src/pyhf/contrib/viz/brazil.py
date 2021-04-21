"""Brazil Band Plots."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.container import Container


class BrazilBandContainer(Container):
    r"""
    Container for the :obj:`matplotlib.artist` objects for the Brazil Band
    created in a :func:`~pyhf.contrib.viz.brazil.plot_results` plot and
    returned by :func:`~pyhf.contrib.viz.brazil.plot_brazil_band`.

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


class ClsComponentsContainer(Container):
    r"""
    Container for the :obj:`matplotlib.artist` objects for the
    :math:`\mathrm{CL}_{s+b}` and :math:`\mathrm{CL}_{b}` lines
    optionally created in a :func:`~pyhf.contrib.viz.brazil.plot_results` plot.

    The container can be treated like a :obj:`collections.namedtuple`
    ``(clsb, clb)``.

    Attributes:
        clsb (:class:`matplotlib.lines.Line2D`): The artist of the
         :math:`\mathrm{CL}_{s+b}` line.

        clb (:class:`matplotlib.lines.Line2D`): The artist of the
         :math:`\mathrm{CL}_{b}` line.
    """

    def __init__(self, cls_components_artists, **kwargs):
        r"""
        Args:
            cls_components_artists (:obj:`tuple`): Tuple of ``(clsb, clb)``.

              * ``clsb`` contains the :class:`matplotlib.lines.Line2D` of the
                observed :math:`\mathrm{CL}_{s+b}` line.

              * ``clb`` contains the :class:`matplotlib.lines.Line2D` of the
                observed :math:`\mathrm{CL}_{b}` line.
        """
        clsb, clb = cls_components_artists
        self.clsb = clsb
        self.clb = clb
        super().__init__(cls_components_artists, **kwargs)


class ResultsPlotContainer(Container):
    r"""
    Container for the :obj:`matplotlib.artist` objects created in a
    :func:`~pyhf.contrib.viz.brazil.plot_results` plot.

    The container can be treated like a :obj:`collections.namedtuple`
    ``(brazil_band, cls_components)``.

    Attributes:
        brazil_band (:class:`BrazilBandContainer`): The container for the Brazil
         band artists.

        cls_components (:class:`ClsComponentsContainer`): The container for the
         artists for the components of the :math:`\mathrm{CL}_{s}` ratio.
    """

    def __init__(self, results_plot_artists, **kwargs):
        r"""
        Args:
            results_plot_artists (:obj:`tuple`): Tuple of
             ``(brazil_band, cls_components)``.

              * ``brazil_band`` contains the :class:`BrazilBandContainer` of the
                Brazil band artists.

              * ``cls_components`` contains the :class:`ClsComponentsContainer` of the
                :math:`\mathrm{CL}_{s}` ratio component artists.
        """
        brazil_band, cls_components = results_plot_artists
        self.brazil_band = brazil_band
        self.cls_components = cls_components
        super().__init__(results_plot_artists, **kwargs)


def plot_brazil_band(mutests, cls_obs, cls_exp, test_size, ax, **kwargs):
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

    return BrazilBandContainer(
        (
            cls_obs_line,
            cls_exp_lines,
            one_sigma_band,
            two_sigma_band,
            test_size_line,
        )
    )


def plot_cls_components(mutests, tail_probs, ax, **kwargs):
    r"""
    Plot the values of :math:`\mathrm{CL}_{s+b}` and :math:`\mathrm{CL}_{b}`
    --- the components of the :math:`\mathrm{CL}_{s}` ratio --- for a series of
    hypothesis tests for various POI values.

    Example:

        :func:`plot_cls_components` is generally meant to be used inside
        :func:`~pyhf.contrib.viz.brazil.plot_results` but can be used by itself.

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
        >>> tail_probs = np.array([test[1] for test in results])
        >>> fig, ax = plt.subplots()
        >>> artists = pyhf.contrib.viz.brazil.plot_cls_components(poi_vals, tail_probs, ax)

    Args:
        mutests (:obj:`list` or :obj:`array`): The values of the POI where the
         hypothesis tests were performed.
        tail_probs (:obj:`list` or :obj:`array`): The values of
         :math:`\mathrm{CL}_{s+b}` and :math:`\mathrm{CL}_{b}` for the POIs
         tested in ``mutests``.
        ax (:obj:`matplotlib.axes.Axes`): The matplotlib axis object to plot on.
        Keywords:
         * ``no_clb`` (:obj:`bool`): Bool for not plotting the
           :math:`\mathrm{CL}_{b}` component.

         * ``no_clsb`` (:obj:`bool`): Bool for not plotting the
           :math:`\mathrm{CL}_{s+b}` component.

    Returns:
        :class:`ClsComponentsContainer`: A container of the
        :obj:`matplotlib.artist` objects drawn.
    """
    clsb_obs = np.array([tail_prob[0] for tail_prob in tail_probs])
    clb_obs = np.array([tail_prob[1] for tail_prob in tail_probs])

    linewidth = kwargs.pop("linewidth", 2)
    no_clsb = kwargs.pop("no_clsb", False)
    no_clb = kwargs.pop("no_clb", False)

    clsb_obs_line_artist = None
    if not no_clsb:
        clsb_color = kwargs.pop("clsb_color", "red")
        clsb_obs_line_artist = ax.plot(
            mutests,
            clsb_obs,
            color=clsb_color,
            linewidth=linewidth,
            label=r"$\mathrm{CL}_{s+b}$",
        )

    clb_obs_line_artist = None
    if not no_clb:
        clb_color = kwargs.pop("clb_color", "blue")
        clb_obs_line_artist = ax.plot(
            mutests,
            clb_obs,
            color=clb_color,
            linewidth=linewidth,
            label=r"$\mathrm{CL}_{b}$",
        )

    return ClsComponentsContainer((clsb_obs_line_artist, clb_obs_line_artist))


def plot_results(mutests, tests, test_size=0.05, ax=None, **kwargs):
    r"""
    Plot a series of hypothesis tests for various POI values.
    For more detail on use of keywords see
    :func:`~pyhf.contrib.viz.brazil.plot_brazil_band` and
    :func:`~pyhf.contrib.viz.brazil.plot_cls_components`.

    Example:

        A Brazil band plot.

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
        >>> artists = pyhf.contrib.viz.brazil.plot_results(poi_vals, results, ax=ax)

        A Brazil band plot with the components of the :math:`\mathrm{CL}_{s}`
        ratio drawn on top.

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
        >>> artists = pyhf.contrib.viz.brazil.plot_results(poi_vals, results, ax=ax, components=True)

    Args:
        mutests (:obj:`list` or :obj:`array`): The values of the POI where the
          hypothesis tests were performed.
        tests (:obj:`list` or :obj:`array`): The collection of :math:`p`-value-like
          values (:math:`\mathrm{CL}_{s}` values or tail probabilities) from
          the hypothesis tests.
          If the ``components`` keyword argument is ``True``,  ``tests`` is required
          to have the same structure as :func:`pyhf.infer.hypotest`'s return
          when using ``return_expected_set=True`` and ``return_tail_probs=True``:
          a tuple of :math:`\mathrm{CL}_{s}`,
          :math:`\left[\mathrm{CL}_{s+b}, \mathrm{CL}_{b}\right]`,
          :math:`\mathrm{CL}_{s,\mathrm{exp}}` band.
        test_size (:obj:`float`): The size, :math:`\alpha`, of the test.
        ax (:obj:`matplotlib.axes.Axes`): The matplotlib axis object to plot on.

    Returns:
        :obj:`~pyhf.contrib.viz.brazil.ResultsPlotContainer`: A container of the
        :obj:`matplotlib.artist` objects drawn.
    """
    if ax is None:
        ax = plt.gca()

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

    brazil_band_container = None
    if not no_cls:
        brazil_band_container = plot_brazil_band(
            mutests, cls_obs, cls_exp, test_size, ax, **kwargs
        )

    x_label = kwargs.pop("xlabel", r"$\mu$ (POI)")
    y_label = kwargs.pop("ylabel", r"$\mathrm{CL}_{s}$")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if ax.get_yscale() == "log":
        ax.set_ylim(test_size * 0.1, 1)
    else:
        ax.set_ylim(0, 1)

    cls_components_container = None
    if plot_components:
        cls_components_container = plot_cls_components(
            mutests, tail_probs, ax, **kwargs
        )

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

    return ResultsPlotContainer((brazil_band_container, cls_components_container))
