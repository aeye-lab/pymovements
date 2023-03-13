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

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import polars as pl
from polars import ColumnNotFoundError

from pymovements.events.event_processing import EventProcessor
from pymovements.events.events import EventDataFrame


def main_sequence_plot(
        data: pl.DataFrame,
        figsize: tuple[int, int] = (15, 5),
        title: str | None = None,
        savepath: str | None = None,
        show: bool = True,
) -> None:
    """
    Plots the saccade main sequence

    Parameters
    ----------
    data:
        Data frame
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


    Exmples
    -------

    """

    try:
        peak_velocities = data['peak_velocity'].to_list()
    except ColumnNotFoundError:
        warnings.warn("The input dataframe you provided does not contain "
                      "the saccade peak velocities. "
                      "We are trying to compute them now.")

        try:
            # check whether the columns we need to calculate the peak velocities
            # are in the df
            _ = data['x_vel', 'y_vel']
            events = EventDataFrame(data)
            processor = EventProcessor('peak_velocity')

            property_result = processor.process(events)
            peak_velocities = property_result['peak_velocity'].to_list()

        except ColumnNotFoundError:
            raise Warning()

    try:
        amplitudes = data['amplitude'].to_list()
    except ColumnNotFoundError:
        warnings.warn("The input dataframe you provided does not contain "
                      "the saccade amplitudes. "
                      "We are trying to compute them now.")

        try:
            # check whether the columns we need to calculate the amplitude
            # are in the df
            _ = data['x_pos', 'y_pos']
            events = EventDataFrame(data)
            processor = EventProcessor('amplitude')

            property_result = processor.process(events)
            amplitudes = property_result['amplitude'].to_list()

        except ColumnNotFoundError:
            raise Warning()

    fig = plt.figure(figsize=figsize)

    plt.scatter(amplitudes, peak_velocities)

    plt.title(title)

    if savepath is not None:
        fig.savefig(savepath)

    if show:
        plt.show()
    plt.close(fig)
