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
"""Module for py:func:`pymovements.gaze.transforms.pos2vel`"""
from __future__ import annotations

import polars as pl

from pymovements.gaze.transforms_pl.savitzky_golay import savitzky_golay
from pymovements.gaze.transforms_pl.transforms_library import register_transform


@register_transform
def pos2vel(
        *,
        sampling_rate: float,
        method: str,
        degree: int | None = None,
        window_length: int | None = None,
        padding: str | float | int | None = 'nearest',
) -> pl.Expr:
    """Compute velocitiy data from positional data.

    There are three methods available for velocity calculation:
    * ``savitzky_golay``: velocity is calculated by a polynomial of fixed degree and window length.
    See :py:func:`~pymovements.gaze.transforms.savitzky_golay` for further details.
    * ``smooth``: velocity is calculated from the difference of the mean values
    of the subsequent two samples and the preceding two samples
    * ``neighbors``: velocity is calculated from difference of the subsequent
    sample and the preceding sample
    * ``preceding``: velocity is calculated from the difference of the current
    sample to the preceding sample

    Parameters
    ----------
    sampling_rate:
        Sampling rate of input time series.
    method:
        The method to use for velocity calculation.
    degree:
        The degree of the polynomial to use. This has only an effect if using ``savitzky_golay`` as
        calculation method.
    window_length:
        The window size to use. This has only an effect if using ``savitzky_golay`` as calculation
        method.
    padding:
        The padding to use.  This has only an effect if using ``savitzky_golay`` as calculation
        method.
    """

    if method == 'neighbors':
        preceding_neighbor = pl.all().shift(periods=1)
        succeeding_neighbor = pl.all().shift(periods=-1)
        return (succeeding_neighbor - preceding_neighbor) * (sampling_rate / 2)

    if method == 'preceding':
        return pl.all().diff(n=1, null_behavior='ignore') * sampling_rate

    if method == 'smooth':
        # Center of window is period 0 and will be filled.
        # mean(arr_-2, arr_-1) and mean(arr_1, arr_2) needs division by two
        # window is now 3 samples long (arr_-1.5, arr_0, arr_1+5)
        # we therefore need a divison by three, all in all it's a division by 6
        return (
            pl.all().shift(periods=-2) + pl.all().shift(periods=-1)
            - pl.all().shift(periods=1) - pl.all().shift(periods=2)
        ) * (sampling_rate / 6)

    if method == 'savitzky_golay':
        if window_length is None:
            raise TypeError("'window_length' must not be none for method 'savitzky_golay'")
        if degree is None:
            raise TypeError("'degree' must not be none for method 'savitzky_golay'")

        return savitzky_golay(
            window_length=window_length,
            degree=degree,
            sampling_rate=sampling_rate,
            padding=padding,
            derivative=1,
        )

    supported_methods = ['preceding', 'neighbors', 'smooth', 'savitzky_golay']
    raise ValueError(
        f"Unknown method '{method}'. Supported methods are: {supported_methods}",
    )
