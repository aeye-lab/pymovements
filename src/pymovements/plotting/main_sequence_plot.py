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
"""
This module holds the main sequence plot.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import polars as pl
from polars import ColumnNotFoundError


def main_sequence_plot(
        data: pl.DataFrame,
        figsize: tuple[int, int] = (15, 5),
        title: str | None = None,
        savepath: str | None = None,
        show: bool = True,
) -> None:
    """
    Plots the saccade main sequence.

    Parameters
    ----------
    data:
        Data frame. It must contain columns "peak_velocity" and "amplitude".
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
    KeyError
        If the input dataframe has no 'amplitude' and/or 'peak_velocity' column.
        Those are needed to create the plot.
    """

    try:
        peak_velocities = data['peak_velocity'].to_list()
    except ColumnNotFoundError as exc:
        raise KeyError(
            'The input dataframe you provided does not contain '
            'the saccade peak velocities which are needed to create '
            'the main sequence plot. ',
        ) from exc

    try:
        amplitudes = data['amplitude'].to_list()
    except ColumnNotFoundError as exc:
        raise KeyError(
            'The input dataframe you provided does not contain '
            'the saccade amplitudes which are needed to create '
            'the main sequence plot. ',
        ) from exc

    fig = plt.figure(figsize=figsize)

    plt.scatter(amplitudes, peak_velocities, color='purple', alpha=0.5)

    plt.title(title)
    plt.xlabel('Amplitude [dva]')
    plt.ylabel('Peak Velocity [dva/s]')

    if savepath is not None:
        fig.savefig(savepath)

    if show:
        plt.show()

    plt.close(fig)
