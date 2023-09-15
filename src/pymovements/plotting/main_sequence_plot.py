# Copyright (c) 2023 The pymovements Project Authors
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
"""This module holds the main sequence plot."""
from __future__ import annotations

import matplotlib.pyplot as plt
import polars as pl
from matplotlib.collections import Collection
from polars import ColumnNotFoundError

from pymovements.events import EventDataFrame


def main_sequence_plot(
        event_df: EventDataFrame,
        marker_size: float = 25,
        color: str = 'purple',
        alpha: float = 0.5,
        marker: str = 'o',
        figsize: tuple[int, int] = (15, 5),
        title: str | None = None,
        savepath: str | None = None,
        show: bool = True,
        **kwargs: Collection,
) -> None:
    """Plot the saccade main sequence.

    Parameters
    ----------
    event_df: Event dataframe
        It must contain columns "peak_velocity" and "amplitude".
    marker_size: float
        Size of the marker symbol.
    color: str
        Color of the marker symbol.
    alpha: float
        Alpha value (=transparency) of the marker symbol. Between 0 and 1.
    marker: str
        Marker symbol. Possible values defined by matplotlib.markers.
    figsize: tuple
        Figure size.
    title: str, optional
        Figure title.
    savepath: str, optional
        If given, figure will be saved to this path.
    show: bool
        If True, figure will be shown.
    **kwargs:
        Additional keyword arguments passed to matplotlib.pyplot.scatter.

    Raises
    ------
    KeyError
        If the input dataframe has no 'amplitude' and/or 'peak_velocity' column.
        Those are needed to create the plot.
    ValueError
        If the event dataframe does not contain any saccades.
    """
    event_col_name = 'name'
    saccades = event_df.frame.filter(pl.col(event_col_name) == 'saccade')

    if saccades.is_empty():
        raise ValueError(
            'There are no saccades in the event dataframe. '
            'Please make sure you ran a saccade detection algorithm. '
            f'The event name should be stored in a colum called "{event_col_name}".',
        )

    try:
        peak_velocities = saccades['peak_velocity'].to_list()
    except ColumnNotFoundError as exc:
        raise KeyError(
            'The input dataframe you provided does not contain '
            'the saccade peak velocities which are needed to create '
            'the main sequence plot. ',
        ) from exc

    try:
        amplitudes = saccades['amplitude'].to_list()
    except ColumnNotFoundError as exc:
        raise KeyError(
            'The input dataframe you provided does not contain '
            'the saccade amplitudes which are needed to create '
            'the main sequence plot. ',
        ) from exc

    fig = plt.figure(figsize=figsize)

    plt.scatter(
        amplitudes,
        peak_velocities,
        color=color,
        alpha=alpha,
        s=marker_size,
        marker=marker,
        # to handle after  https://github.com/pydata/xarray/pull/8030 is merged
        **kwargs,  # type: ignore
    )

    if title:
        plt.title(title)
    plt.xlabel('Amplitude [dva]')
    plt.ylabel('Peak Velocity [dva/s]')

    if savepath is not None:
        fig.savefig(savepath)

    if show:
        plt.show()

    plt.close(fig)
