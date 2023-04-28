# Copyright (c) 2022-2023 The pymovements Project Authors
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
"""
This module holds the time series plot.
"""
from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np

from pymovements.gaze import GazeDataFrame


def tsplot(
        gaze: GazeDataFrame,
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
    """
    Plot time series with each channel getting a separate subplot.

    Parameters
    ----------
    gaze:
        The GazeDataFrame to plot.
    channels: list, optional
        list of channel names
    n_cols: int
        Number of channel subplot colunms. If None, it will be automatically inferred.
    n_rows: int
        Number of channel subplot rows. If None, it will be automatically inferred.
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
    if channels is None:
        channels = gaze.frame.columns

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

    # set ylims to have zero centered y-axis (for all axes)
    if share_y and zero_centered_yaxis:
        y_pad_factor = 1.1
        ylim_abs = np.nanmax(np.abs(arr))
        ylims = -ylim_abs * y_pad_factor, ylim_abs * y_pad_factor

    for channel_id in range(n_channels):
        ax = axs[channel_id]

        x_channel = arr[channel_id, :]
        ax.plot(t, x_channel, color=line_color, linewidth=line_width)

        if not share_y:
            y_pad_factor = 1.1
            ylim_abs = np.nanmax(np.abs(arr[channel_id]))
            ylims = -ylim_abs * y_pad_factor, ylim_abs * y_pad_factor

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
