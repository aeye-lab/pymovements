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

import polars as pl

import pymovements as pm  # pylint: disable=cyclic-import
from pymovements.stimulus.stimulus import Stimulus


class TextStimulus(Stimulus):
    def __init__(
            self,
            aois: pl.DataFrame,
            character_column,
            pixel_x_column,
            pixel_y_column,
            width_column,
            height_column,
            page_column,
    ):
        if aois is None:
            aois = pl.DataFrame()
        else:
            aois = aois.clone()
        self.frame = aois
        self.frame = self.frame.rename(
            {
                character_column: 'character',
                pixel_x_column: 'top_left_x',
                pixel_y_column: 'top_left_y',
                width_column: 'aoi_width',
                height_column: 'aoi_height',
                page_column: 'page',
            },
        )


def from_file(
        aoi_path: str | Path,
        *,
        character_column: str,
        pixel_x_column: str,
        pixel_y_column: str,
        width_column: str,
        height_column: str,
        page_column: str,
        custom_read_kwargs: dict[str, Any] | None = None,
) -> TextStimulus:
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
            f'Supported formats are: {valid_extensions}',
        )


    text_stimulus = TextStimulus(
        stimulus_df,
        character_column,
        pixel_x_column,
        pixel_y_column,
        width_column,
        height_column,
        page_column,
    )
    return text_stimulus
