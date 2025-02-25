# Copyright (c) 2024-2025 The pymovements Project Authors
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
from pymovements.utils import checks


def get_aoi(
        aoi_dataframe: TextStimulus,
        row: pl.DataFrame.row,
        x_eye: str,
        y_eye: str,
) -> str:
    """Given eye movement and aoi dataframe, return aoi.

    If `width` is used, calculation: start_x_column <= x_eye < start_x_column + width.
    If `end_x_column` is used, calculation: start_x_column <= x_eye < end_x_column.
    Analog for y coordinate and height.

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

    Raises
    ------
    ValueError
        If width and end_TYPE_column is None.
    """
    if aoi_dataframe.width_column is not None:
        checks.check_is_none_is_mutual(
            height_column=aoi_dataframe.width_column,
            width_column=aoi_dataframe.height_column,
        )
        aoi = aoi_dataframe.aois.filter(
            (aoi_dataframe.aois[aoi_dataframe.start_x_column] <= row[x_eye]) &
            (
                row[x_eye] <
                aoi_dataframe.aois[aoi_dataframe.start_x_column] +
                aoi_dataframe.aois[aoi_dataframe.width_column]
            ) &
            (aoi_dataframe.aois[aoi_dataframe.start_y_column] <= row[y_eye]) &
            (
                row[y_eye] <
                aoi_dataframe.aois[aoi_dataframe.start_y_column] +
                aoi_dataframe.aois[aoi_dataframe.height_column]
            ),
        )

        if aoi.is_empty():
            aoi.extend(pl.from_dict({col: None for col in aoi.columns}))
        return aoi
    if aoi_dataframe.end_x_column is not None:
        checks.check_is_none_is_mutual(
            end_x_column=aoi_dataframe.end_x_column,
            end_y_column=aoi_dataframe.end_y_column,
        )
        aoi = aoi_dataframe.aois.filter(
            # x-coordinate: within bounding box
            (aoi_dataframe.aois[aoi_dataframe.start_x_column] <= row[x_eye]) &
            (row[x_eye] < aoi_dataframe.aois[aoi_dataframe.end_x_column]) &
            # y-coordinate: within bounding box
            (aoi_dataframe.aois[aoi_dataframe.start_y_column] <= row[y_eye]) &
            (row[y_eye] < aoi_dataframe.aois[aoi_dataframe.end_y_column]),
        )

        if aoi.is_empty():
            aoi.extend(pl.from_dict({col: None for col in aoi.columns}))

        return aoi
    raise ValueError(
        'either aoi_dataframe.width or aoi_dataframe.end_x_column have to be not None',
    )
