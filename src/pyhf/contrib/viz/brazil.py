"""Brazil Band Plots."""
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    "BrazilBandCollection",
    "plot_brazil_band",
    "plot_cls_components",
    "plot_results",
]


def __dir__():
    return __all__


class BrazilBandCollection(
    namedtuple(
        "BrazilBandCollection",
        (
            "cls_obs",
            "cls_exp",
            "one_sigma_band",
            "two_sigma_band",
            "test_size",
            "clsb",
            "clb",
            "axes",
        ),
    )
):
    r"""
    :obj:`collections.namedtuple` containing the
    :class:`matplotlib.artist.Artist` objects of the
    "Brazil Band" and the observed :math:`\mathrm{CL}_{s+b}` and
    :math:`\mathrm{CL}_{b}` --- the components of the
    :math:`\mathrm{CL}_{s}` ratio.
    Returned by :func:`~pyhf.contrib.viz.brazil.plot_results`.

    :param cls_obs: The :class:`matplotlib.lines.Line2D` of the
     :math:`\mathrm{CL}_{s,\mathrm{obs}}` line.

    :param cls_exp: The :obj:`list` of :class:`matplotlib.lines.Line2D` of the
     :math:`\mathrm{CL}_{s,\mathrm{exp}}` lines.

    :param one_sigma_band: The :class:`matplotlib.collections.PolyCollection` of
     the :math:`\mathrm{CL}_{s,\mathrm{exp}}` :math:`\pm1\sigma` band.

    :param two_sigma_band: The :class:`matplotlib.collections.PolyCollection` of
     the :math:`\mathrm{CL}_{s,\mathrm{exp}}` :math:`\pm2\sigma` band.

    :param test_size: The :class:`matplotlib.lines.Line2D` of the test size line.

    :param clsb: The :class:`matplotlib.lines.Line2D` of the observed
     :math:`\mathrm{CL}_{s+b}` line.

    :param clb: The :class:`matplotlib.lines.Line2D` of the observed
     :math:`\mathrm{CL}_{b}` line.

    :param axes: The :class:`matplotlib.axes.Axes` the artists are plotted on.
    """


def plot_brazil_band(test_pois, cls_obs, cls_exp, test_size, ax, **kwargs):
    r"""
    Plot the values of :math:`\mathrm{CL}_{s,\mathrm{obs}}` and the
    :math:`\mathrm{CL}_{s,\mathrm{exp}}` band (the "Brazil band") for a series
    of hypothesis tests for various POI values.

    Example:

        :func:`plot_brazil_band` is generally meant to be used inside
        :func:`~pyhf.contrib.viz.brazil.plot_results` but can be used by itself.

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> import pyhf
        >>> import pyhf.contrib.viz.brazil
        >>> pyhf.set_backend("numpy")
        >>> model  = pyhf.simplemodels.uncorrelated_background(
        ...     signal=[12.0, 11.0], bkg=[50.0, 52.0], bkg_uncertainty=[3.0, 7.0]
        ... )
        >>> observations = [51, 48]
        >>> data = observations + model.config.auxdata
        >>> test_pois = np.linspace(0, 5, 41)
        >>> results = [
        ...     pyhf.infer.hypotest(test_poi, data, model, return_expected_set=True)
        ...     for test_poi in test_pois
        ... ]
        >>> cls_obs = np.array([test[0] for test in results]).flatten()
        >>> cls_exp = [
        ...     np.array([test[1][sigma_idx] for test in results]).flatten()
        ...     for sigma_idx in range(5)
        ... ]
        >>> test_size = 0.05
        >>> fig, ax = plt.subplots()
        >>> artists = pyhf.contrib.viz.brazil.plot_brazil_band(
        ...     test_pois, cls_obs, cls_exp, test_size, ax
        ... )

    Args:
        test_pois (:obj:`list` or :obj:`array`): The values of the POI where the
         hypothesis tests were performed.
        cls_obs (:obj:`list` or :obj:`array`): The values of
         :math:`\mathrm{CL}_{s,\mathrm{obs}}` for the POIs tested in ``test_pois``.
        cls_exp (:obj:`list` or :obj:`array`): The values of the
         :math:`\mathrm{CL}_{s,\mathrm{exp}}` band for the POIs tested in
         ``test_pois``.
        test_size (:obj:`float`): The size, :math:`\alpha`, of the test.
        ax (:obj:`matplotlib.axes.Axes`): The matplotlib axis object to plot on.

    Returns:
        :obj:`tuple`: The :obj:`matplotlib.artist` objects drawn.
    """
    line_color = kwargs.pop("color", "black")
    (cls_obs_line,) = ax.plot(
        test_pois, cls_obs, color=line_color, label=r"$\mathrm{CL}_{s}$"
    )

    cls_exp_lines = []
    for idx, color in zip(range(5), 5 * [line_color]):
        (_cls_exp_line,) = ax.plot(
            test_pois,
            cls_exp[idx],
            color=color,
            linestyle="dotted" if idx != 2 else "dashed",
            label=None if idx != 2 else r"$\mathrm{CL}_{s,\mathrm{exp}}$",
        )
        cls_exp_lines.append(_cls_exp_line)
    one_sigma_band = ax.fill_between(
        test_pois,
        cls_exp[0],
        cls_exp[-1],
        facecolor="yellow",
        label=r"$\pm2\sigma$ $\mathrm{CL}_{s,\mathrm{exp}}$",
    )
    two_sigma_band = ax.fill_between(
        test_pois,
        cls_exp[1],
        cls_exp[-2],
        facecolor="green",
        label=r"$\pm1\sigma$ $\mathrm{CL}_{s,\mathrm{exp}}$",
    )

    test_size_color = kwargs.pop("test_size_color", "red")
    test_size_linestyle = kwargs.pop("test_size_linestyle", "solid")
    (test_size_line,) = ax.plot(
        test_pois,
        [test_size] * len(test_pois),
        color=test_size_color,
        linestyle=test_size_linestyle,
        label=rf"$\alpha={test_size}$",
    )

    return cls_obs_line, cls_exp_lines, one_sigma_band, two_sigma_band, test_size_line


def plot_cls_components(test_pois, tail_probs, ax, **kwargs):
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
        >>> model = pyhf.simplemodels.uncorrelated_background(
        ...     signal=[12.0, 11.0], bkg=[50.0, 52.0], bkg_uncertainty=[3.0, 7.0]
        ... )
        >>> observations = [51, 48]
        >>> data = observations + model.config.auxdata
        >>> test_pois = np.linspace(0, 5, 41)
        >>> results = [
        ...     pyhf.infer.hypotest(
        ...         test_poi, data, model, return_expected_set=True, return_tail_probs=True
        ...     )
        ...     for test_poi in test_pois
        ... ]
        >>> tail_probs = np.array([test[1] for test in results])
        >>> fig, ax = plt.subplots()
        >>> artists = pyhf.contrib.viz.brazil.plot_cls_components(test_pois, tail_probs, ax)

    Args:
        test_pois (:obj:`list` or :obj:`array`): The values of the POI where the
         hypothesis tests were performed.
        tail_probs (:obj:`list` or :obj:`array`): The values of
         :math:`\mathrm{CL}_{s+b}` and :math:`\mathrm{CL}_{b}` for the POIs
         tested in ``test_pois``.
        ax (:obj:`matplotlib.axes.Axes`): The matplotlib axis object to plot on.
        Keywords:
         * ``no_clb`` (:obj:`bool`): Bool for not plotting the
           :math:`\mathrm{CL}_{b}` component.

         * ``no_clsb`` (:obj:`bool`): Bool for not plotting the
           :math:`\mathrm{CL}_{s+b}` component.

    Returns:
        :obj:`tuple`: The :obj:`matplotlib.lines.Line2D` artists drawn.
    """
    clsb_obs = np.array([tail_prob[0] for tail_prob in tail_probs])
    clb_obs = np.array([tail_prob[1] for tail_prob in tail_probs])

    linewidth = kwargs.pop("linewidth", 2)
    no_clsb = kwargs.pop("no_clsb", False)
    no_clb = kwargs.pop("no_clb", False)

    clsb_obs_line_artist = None
    if not no_clsb:
        clsb_color = kwargs.pop("clsb_color", "red")
        (clsb_obs_line_artist,) = ax.plot(
            test_pois,
            clsb_obs,
            color=clsb_color,
            linewidth=linewidth,
            label=r"$\mathrm{CL}_{s+b}$",
        )

    clb_obs_line_artist = None
    if not no_clb:
        clb_color = kwargs.pop("clb_color", "blue")
        (clb_obs_line_artist,) = ax.plot(
            test_pois,
            clb_obs,
            color=clb_color,
            linewidth=linewidth,
            label=r"$\mathrm{CL}_{b}$",
        )

    return clsb_obs_line_artist, clb_obs_line_artist


def plot_results(test_pois, tests, test_size=0.05, ax=None, **kwargs):
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
        >>> model = pyhf.simplemodels.uncorrelated_background(
        ...     signal=[12.0, 11.0], bkg=[50.0, 52.0], bkg_uncertainty=[3.0, 7.0]
        ... )
        >>> observations = [51, 48]
        >>> data = observations + model.config.auxdata
        >>> test_pois = np.linspace(0, 5, 41)
        >>> results = [
        ...     pyhf.infer.hypotest(test_poi, data, model, return_expected_set=True)
        ...     for test_poi in test_pois
        ... ]
        >>> fig, ax = plt.subplots()
        >>> artists = pyhf.contrib.viz.brazil.plot_results(test_pois, results, ax=ax)

        A Brazil band plot with the components of the :math:`\mathrm{CL}_{s}`
        ratio drawn on top.

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> import pyhf
        >>> import pyhf.contrib.viz.brazil
        >>> pyhf.set_backend("numpy")
        >>> model = pyhf.simplemodels.uncorrelated_background(
        ...     signal=[12.0, 11.0], bkg=[50.0, 52.0], bkg_uncertainty=[3.0, 7.0]
        ... )
        >>> observations = [51, 48]
        >>> data = observations + model.config.auxdata
        >>> test_pois = np.linspace(0, 5, 41)
        >>> results = [
        ...     pyhf.infer.hypotest(
        ...         test_poi, data, model, return_expected_set=True, return_tail_probs=True
        ...     )
        ...     for test_poi in test_pois
        ... ]
        >>> fig, ax = plt.subplots()
        >>> artists = pyhf.contrib.viz.brazil.plot_results(
        ...     test_pois, results, ax=ax, components=True
        ... )

    Args:
        test_pois (:obj:`list` or :obj:`array`): The values of the POI where the
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
        :class:`BrazilBandCollection`: Artist containing the :obj:`matplotlib.artist`
        objects drawn.
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

    brazil_band_artists = (
        plot_brazil_band(test_pois, cls_obs, cls_exp, test_size, ax, **kwargs)
        if not no_cls
        else (None,) * 5
    )

    clsb, clb = (
        plot_cls_components(test_pois, tail_probs, ax, **kwargs)
        if plot_components
        else (None, None)
    )

    x_label = kwargs.pop("xlabel", r"$\mu$ (POI)")
    y_label = kwargs.pop("ylabel", r"$\mathrm{CL}_{s}$")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if ax.get_yscale() == "log":
        ax.set_ylim(test_size * 0.1, 1)
    else:
        ax.set_ylim(0, 1)

    # Order legend: ensure CLs expected band and test size are last in legend
    handles, labels = ax.get_legend_handles_labels()
    if not no_cls:
        for label_part in ["exp", "pm1", "pm2", "alpha"]:
            label_idx = next(
                idx for idx, label in enumerate(labels) if label_part in label
            )
            handles.append(handles.pop(label_idx))
            labels.append(labels.pop(label_idx))

    ax.legend(handles, labels, loc="best")

    return BrazilBandCollection(*brazil_band_artists, clsb, clb, ax)
