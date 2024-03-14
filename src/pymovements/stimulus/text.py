# Copyright (c) 2023-2024 The pymovements Project Authors
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
"""Module for the TextDataFrame."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl


class TextStimulus:
    """A DataFrame for gaze time series data.

    Each row is a sample at a specific timestep.
    Each column is a channel in the gaze time series.

    Parameters
    ----------
    aois: pl.DataFrame
        A stimulus dataframe.
    aoi_column: str
        Name of the column that contains the content of the aois.
    pixel_x_column: str
        Name of the column which contains the x coordinate of the areas of interest.
    pixel_y_column: str
        Name of the column which contains the y coordinate of the areas of interest.
    width_column: str
        Name of the column which contains the width of the area of interest.
    height_column: str
        Name of the column which contains the height of the area of interest.
    page_column: str
        Name of the column which contains the page information of the area of interest.
    """

    def __init__(
            self,
            aois: pl.DataFrame,
            *,
            aoi_column: str,
            pixel_x_column: str,
            pixel_y_column: str,
            width_column: str,
            height_column: str,
            page_column: str,
    ) -> None:

        self.aois = aois.clone()
        self.aoi_column = aoi_column
        self.pixel_x_column = pixel_x_column
        self.pixel_y_column = pixel_y_column
        self.width_column = width_column
        self.height_column = height_column
        self.page_column = page_column


def from_file(
        aoi_path: str | Path,
        *,
        aoi_column: str,
        pixel_x_column: str,
        pixel_y_column: str,
        width_column: str,
        height_column: str,
        page_column: str,
        custom_read_kwargs: dict[str, Any] | None = None,
) -> TextStimulus:
    """Load text stimulus from file.

    Parameters
    ----------
    aoi_path:  str | Path
        Path to file to be read.
    aoi_column: str
        Name of the column that contains the content of the aois.
    pixel_x_column: str
        Name of the column which contains the x coordinate of the areas of interest.
    pixel_y_column: str
        Name of the column which contains the y coordinate of the areas of interest.
    width_column: str
        Name of the column which contains the width of the area of interest.
    height_column: str
        Name of the column which contains the height of the area of interest.
    page_column: str
        Name of the column which contains the page information of the area of interest.
    custom_read_kwargs: dict[str, Any] | None
        Custom read keyword arguments for polars. (default: None)

    Returns
    -------
    TextStimulus
        Returns the text stimulus file.
    """
    if isinstance(aoi_path, str):
        aoi_path = Path(aoi_path)
    if custom_read_kwargs is None:
        custom_read_kwargs = {}

    valid_extensions = {'.csv', '.tsv', '.txt'}
    if aoi_path.suffix in valid_extensions:
        stimulus_df = pl.read_csv(
            aoi_path,
            **custom_read_kwargs,
        )
    else:
        raise ValueError(
            f'unsupported file format "{aoi_path.suffix}".'
            f'Supported formats are: {sorted(valid_extensions)}',
        )

    return TextStimulus(
        aois=stimulus_df,
        aoi_column=aoi_column,
        pixel_x_column=pixel_x_column,
        pixel_y_column=pixel_y_column,
        width_column=width_column,
        height_column=height_column,
        page_column=page_column,
    )
