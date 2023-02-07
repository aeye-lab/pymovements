"""
This module holds all plotting functions.
"""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.collections import LineCollection


default_segmentdata = {
    'red':   [
        [0.0,  0.0, 0.0],
        [0.5,  1.0, 1.0],
        [1.0,  1.0, 1.0],
    ],
    'green': [
        [0.0,  0.0, 0.0],
        [0.5,  1.0, 1.0],
        [1.0,  0.0, 0.0],
    ],
    'blue':  [
        [0.0,  0.0, 0.0],
        [0.5,  0.0, 0.0],
        [1.0,  0.0, 0.0],
    ],
}


default_segmentdata_twoslope = {
    'red':   [
        [0.0,  0.0, 0.0],
        [0.5,  0.0, 0.0],
        [0.75, 1.0, 1.0],
        [1.0,  1.0, 1.0],
    ],
    'green': [
        [0.0,  0.0, 0.0],
        [0.25, 1.0, 1.0],
        [0.5,  0.0, 0.0],
        [0.75, 1.0, 1.0],
        [1.0,  0.0, 0.0],
    ],
    'blue':  [
        [0.0,  1.0, 1.0],
        [0.25, 1.0, 1.0],
        [0.5,  0.0, 0.0],
        [1.0,  0.0, 0.0],
    ],
}


def traceplot(
    x: np.ndarray,
    y: np.ndarray,
    cval: np.ndarray | None = None,
    cmap: colors.Colormap | None = None,
    cmap_norm: colors.Normalize | str | None = None,
    cmap_segmentdata: dict[str, list[list[float]]] | None = None,
    cbar_label: str | None = None,
    show_cbar: bool = False,
    padding: float | None = None,
    pad_factor: float | None = 0.05,
    figsize: tuple[int, int] = (15, 5),
    title: str | None = None,
    savepath: str | None = None,
    show: bool = True,
) -> None:
    """
    Plot eye gaze trace from positional data.

    Parameters
    ----------
    x: np.ndarray
        x-coordinates
    y: np.ndarray
        y-coordinates
    cval: np.ndarray
        line color values.
    cmap: matplotlib.colors.Colormap, optional
        color map for line color values
    cmap_norm: matplotlib.colors.Normalize, str, optional
        normalization for color values
    cmap_segmentdata: dict, optional
        color map segmentation to build color map
    cbar_label: str, optional
        string label for color bar
    show_cbar: bool
        Shows color bar if True.
    padding: float, optional
        Absolute padding value. If None it is inferred from pad_factor and limits.
    pad_factor: float, optional
        Relative padding factor to construct padding value if not given.
    figsize: tuple
        Figure size.
    title: str, optional
        Figure title.
    savepath: str, optional
        If given, figure will be saved to this path.
    show: bool
        If True, figure will be shown.

    Raises
    ------
    ValueError
        If length of x and y coordinates do not match.
    """

    if len(x) != len(y):
        raise ValueError(
            'x and y do not share same length '
            f'({len(x)} != {len(y)})',
        )
    n = len(x)

    fig = plt.figure(figsize=figsize)
    ax = fig.gca()

    if cval is None:
        cval = np.zeros(n)
        show_cbar = False

    if cmap_norm is None:
        cval_max = np.nanmax(np.abs(cval))
        cval_min = np.nanmin(cval)

        if cval_max and cval_min < 0:
            cmap_norm = 'twoslope'
        elif cval_max:
            cmap_norm = 'normalize'
        else:
            cmap_norm = 'nonorm'

    if cmap is None:
        if cmap_segmentdata is None:
            if cmap_norm == 'twoslope':
                cmap_segmentdata = default_segmentdata_twoslope
            else:
                cmap_segmentdata = default_segmentdata

        cmap = matplotlib.colors.LinearSegmentedColormap(
            'line_cmap', segmentdata=cmap_segmentdata, N=512,
        )

    if cmap_norm == 'twoslope':
        cmap_norm = matplotlib.colors.TwoSlopeNorm(
            vcenter=0, vmin=-cval_max, vmax=cval_max,
        )
    elif cmap_norm == 'normalize':
        cmap_norm = matplotlib.colors.Normalize(
            vmin=cval_min, vmax=cval_max,
        )
    elif cmap_norm == 'nonorm':
        cmap_norm = matplotlib.colors.NoNorm()

    # Create a set of line segments so that we can color them individually
    # This creates the points as a N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be (numlines) x (points per line) x 2 (for x and y)
    points = np.array([x, y]).T.reshape((-1, 1, 2))
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create a continuous norm to map from data points to colors
    line_collection = LineCollection(segments, cmap=cmap, norm=cmap_norm)
    # Set the values used for colormapping
    line_collection.set_array(cval)
    line_collection.set_linewidth(2)
    line = ax.add_collection(line_collection)

    if padding is None:
        x_pad = (np.nanmax(x) - np.nanmin(x)) * pad_factor
        y_pad = (np.nanmax(y) - np.nanmin(y)) * pad_factor
    else:
        x_pad = padding
        y_pad = padding

    ax.set_xlim(np.nanmin(x) - x_pad, np.nanmax(x) + x_pad)
    ax.set_ylim(np.nanmin(y) - y_pad, np.nanmax(y) + y_pad)

    if show_cbar:
        # sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=cmap_norm)
        # sm.set_array(cval)
        fig.colorbar(line, label=cbar_label, ax=ax)

    ax.set_title(title)

    if savepath is not None:
        fig.savefig(savepath)

    if show:
        plt.show()
    plt.close(fig)


def tsplot(
    arr: np.array,
    channel_names: list[str] | None = None,
    xlabel: str | None = None,
    rotate_ylabels: bool = True,
    share_y: bool = True,
    zero_centered_yaxis: bool = True,
    line_color: tuple[int, int, int] | str = 'k',
    line_width: int = 1,
    show_grid: bool = True,
    show_yticks: bool = True,
    figsize: tuple[int, int] = (15, 5),
    title: str | None = None,
    savepath: str | None = None,
    show: bool = True,
) -> None:
    """
    Plot time series with each channel getting a separate subplot.

    Parameters
    ----------
    arr: np.ndarray
        array with channeled time series
    channel_names: list, optional
        list of channel names
    xlabel: str, optional
        set x label
    rotate_ylabels: bool
        set to rotate ylabels
    share_y: bool
        set if y-axes should share common axis
    zero_centered_yaxis: bool
        set if y-axis should be zero centered
    line_color: tuple, str
        set line color
    line_width: int
        set line width
    show_grid: bool
        set to show grid
    show_yticks: bool
        set to show yticks
    figsize: tuple
        Figure size.
    title: str, optional
        Figure title.
    savepath: str, optional
        If given, figure will be saved to this path.
    show: bool
        If True, figure will be shown.

    Raises
    ------
    ValueError
        If array has more than two dimensions.
    """

    if arr.ndim == 1:
        arr = np.expand_dims(arr, axis=0)
    elif arr.ndim == 2:
        pass
    else:
        raise ValueError(arr.shape)

    channel_axis = 0
    sample_axis = 1

    n_channels = arr.shape[channel_axis]
    n_samples = arr.shape[sample_axis]

    # determine number of subplots and height ratios for events
    n_subplots = n_channels
    height_ratios = [1] * n_channels

    fig, axs = plt.subplots(
        nrows=n_subplots,
        sharex=True, sharey=share_y,
        figsize=figsize,
        gridspec_kw={
            'hspace': 0,
            'height_ratios': height_ratios,
        },
    )

    t = np.arange(n_samples)
    xlims = t.min(), t.max()

    # set ylims to have zero centered y-axis (for all axes)
    if share_y and zero_centered_yaxis:
        y_pad_factor = 1.1
        ylim_abs = np.nanmax(np.abs(arr))
        ylims = -ylim_abs * y_pad_factor, ylim_abs * y_pad_factor

    for channel_id in range(n_channels):
        ax = axs[channel_id]
        x_channel = arr[channel_id, :]

        ax.plot(t, x_channel, color=line_color, linewidth=line_width)

        ax.set_xlim(xlims)
        ax.set_ylim(ylims)

        ax.grid(show_grid, which='major')
        ax.grid(show_grid, which='minor')

        ax.tick_params(
            which='both', direction='out',
            length=4, width=1, colors='k',
            grid_color='#999999', grid_alpha=0.5,
        )

        if show_yticks:
            # from matplotlib.ticker import AutoMinorLocator
            # ax.yaxis.set_minor_locator(AutoMinorLocator())
            plt.setp(ax.get_yticklabels(), visible=True)
        else:
            ax.set_yticks([])
            plt.setp(ax.get_yticklabels(), visible=False)

        params = {'mathtext.default': 'regular'}
        plt.rcParams.update(params)

        # set channel names as y-axis labels
        if channel_names:
            if rotate_ylabels:
                ax.set_ylabel(
                    channel_names[channel_id],
                    rotation='horizontal',
                    ha='right', va='center',
                )
            else:
                ax.set_ylabel(channel_names[channel_id])

    # print x label on last used (bottom) axis
    ax.set_xlabel(xlabel)

    axs[0].set_title(title)

    if savepath is not None:
        fig.savefig(savepath)

    if show:
        plt.show()
    plt.close(fig)
