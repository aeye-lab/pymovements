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
    """TextStimulus class."""

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
    """Load text stimulus from file."""
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
