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
"""Module for py:func:`pymovements.gaze.transforms.pos2acc`"""
from __future__ import annotations

import polars as pl

from pymovements.gaze.transforms_pl.savitzky_golay import savitzky_golay
from pymovements.gaze.transforms_pl.transforms_library import register_transform


@register_transform
def pos2acc(
        *,
        sampling_rate: float,
        n_components: int,
        degree: int = 2,
        window_length: int = 7,
        padding: str | float | int | None = 'nearest',
        position_column: str = 'position',
        acceleration_column: str = 'acceleration',
) -> pl.Expr:
    """Compute acceleration data from positional data.

    Parameters
    ----------
    sampling_rate:
        Sampling rate of input time series.
    degree:
        The degree of the polynomial to use.
    window_length:
        The window size to use.
    padding:
        The padding to use.
    n_components:
        Number of components in input column.
    position_column:
        The input position column name.
    acceleration_column:
        The output acceleration column name.
    """
    return savitzky_golay(
        window_length=window_length,
        degree=degree,
        sampling_rate=sampling_rate,
        padding=padding,
        derivative=2,
        n_components=n_components,
        input_column=position_column,
        output_column=acceleration_column,
    )
