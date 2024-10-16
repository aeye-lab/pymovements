# Copyright (c) 2024 The pymovements Project Authors
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
"""Aoi utilities."""
from __future__ import annotations

import polars as pl

from pymovements.stimulus import TextStimulus


def get_aoi(
        aoi_dataframe: TextStimulus,
        row: pl.DataFrame.row,
        x_eye: str,
        y_eye: str,
) -> str:
    """Given eye movement and aoi dataframe, return aoi.

    Parameters
    ----------
    aoi_dataframe: TextStimulus
        Text dataframe to containing area of interests.
    row: pl.DataFrame.row
        Eye movement row.
    x_eye: str
        Name of x eye coordinate.
    y_eye: str
        Name of y eye coordinate.

    Returns
    -------
    str
        Looked at area of interest.
    """
    try:
        aoi = aoi_dataframe.aois.filter(
            (aoi_dataframe.aois[aoi_dataframe.pixel_x_column] <= row[x_eye]) &
            (
                row[x_eye] <
                aoi_dataframe.aois[aoi_dataframe.pixel_x_column] +
                aoi_dataframe.aois[aoi_dataframe.width_column]
            ) &
            (aoi_dataframe.aois[aoi_dataframe.pixel_y_column] <= row[y_eye]) &
            (
                row[y_eye] <
                aoi_dataframe.aois[aoi_dataframe.pixel_y_column] +
                aoi_dataframe.aois[aoi_dataframe.height_column]
            ),
        )[aoi_dataframe.aoi_column].item()
        return aoi
    except ValueError:
        return ''
