# Copyright (c) 2022-2025 The pymovements Project Authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Provides the time series plotting function."""
from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from pymovements.gaze import Gaze


def tsplot(
        gaze: Gaze,
        channels: list[str] | None = None,
        xlabel: str | None = None,
        n_cols: int | None = None,
        n_rows: int | None = None,
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
    """Plot time series with each channel getting a separate subplot.

    Parameters
    ----------
    gaze: Gaze
        The Gaze to plot.
    channels: list[str] | None
        List of channel names to plot. If None, all channels will be plotted. (default: None)
    xlabel: str | None
        Set the x label. (default: None)
    n_cols: int | None
        Number of channel subplot colunms. If None, it will be automatically inferred.
        (default: None)
    n_rows: int | None
        Number of channel subplot rows. If None, it will be automatically inferred. (default: None)
    rotate_ylabels: bool
        Set whether to rotate ylabels. (default: True)
    share_y: bool
        Set if y-axes should share common axis. (default: True)
    zero_centered_yaxis: bool
        Set if y-axis should be zero centered. (default: True)
    line_color: tuple[int, int, int] | str
        Set line color. (default: 'k')
    line_width: int
        Set line width. (default: 1)
    show_grid: bool
        Set whether to show the background grid. (default: True)
    show_yticks: bool
        Set whether to show yticks. (default: True)
    figsize: tuple[int, int]
        Figure size. (default: (15, 5))
    title: str | None
        Figure title. (default: None)
    savepath: str | None
        If given, figure will be saved to this path. (default: None)
    show: bool
        If True, figure will be shown. (default: True)

    Raises
    ------
    ValueError
        If array has more than two dimensions.
    """
    if channels is None:
        channels = [c for c in gaze.frame.columns if gaze.frame[c].dtype != pl.List]

    arr = gaze.frame[channels].to_numpy().transpose()

    if arr.ndim == 1:
        arr = np.expand_dims(arr, axis=0)

    channel_axis = 0
    sample_axis = 1

    n_channels = arr.shape[channel_axis]
    n_samples = arr.shape[sample_axis]

    if n_cols is None:
        if n_channels % 2 == 0:
            n_cols = 2
        else:
            n_cols = 1

    if n_rows is None:
        n_rows = math.ceil(n_channels / n_cols)

    # determine number of subplots and height ratios for events
    height_ratios = [1] * n_rows

    fig, axs = plt.subplots(
        ncols=n_cols,
        nrows=n_rows,
        sharex=True,
        sharey=share_y,
        squeeze=False,
        figsize=figsize,
        gridspec_kw={
            'hspace': 0,
            'height_ratios': height_ratios,
        },
    )
    axs = axs.flatten()

    t = np.arange(n_samples)
    xlims = t.min(), t.max()

    y_pad_factor = 1.1

    # set ylims to have zero centered y-axis (for all axes)
    # will be overwritten if share_y is False
    if zero_centered_yaxis:
        ylim_abs = np.nanmax(np.abs(arr))
        ylims = -ylim_abs * y_pad_factor, ylim_abs * y_pad_factor
    else:
        ylim_max = np.nanmax(arr)
        ylim_min = np.nanmin(arr)
        ylims = ylim_min * y_pad_factor, ylim_max * y_pad_factor

    for channel_id in range(n_channels):
        ax = axs[channel_id]

        x_channel = arr[channel_id, :]
        ax.plot(t, x_channel, color=line_color, linewidth=line_width)

        if not share_y and zero_centered_yaxis:
            ylim_abs = np.nanmax(np.abs(arr[channel_id]))
            ylims = -ylim_abs * y_pad_factor, ylim_abs * y_pad_factor
        elif not share_y and not zero_centered_yaxis:
            ylim_max = np.nanmax(arr)
            ylim_min = np.nanmin(arr)
            ylims = ylim_min * y_pad_factor, ylim_max * y_pad_factor

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
        if rotate_ylabels:
            ax.set_ylabel(
                channels[channel_id],
                rotation='horizontal',
                ha='right', va='center',
            )
        else:
            ax.set_ylabel(channels[channel_id])

    # print x label on last used (bottom) axis
    ax.set_xlabel(xlabel)

    if n_channels == 1:
        ax.set_title(title)
    else:
        axs[0].set_title(title)

    if savepath is not None:
        fig.savefig(savepath)

    if show:
        plt.show()
    plt.close(fig)
