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
"""Module for py:func:`pymovements.gaze.transforms.center_origin`"""
from __future__ import annotations

import polars as pl

from pymovements.gaze.transforms_pl.transforms_library import register_transform


@register_transform
def center_origin(
        *,
        screen_resolution: tuple[int, int],
        origin: str,
        n_components: int,
        pixel_column: str = 'pixel',
        output_column: str | None = None,
) -> pl.Expr:
    """Center pixel data.

    Pixel data will have the coordinates ``(0, 0)`` afterwards.

    Parameters
    ----------
    screen_resolution:
        Pixel screen resolution as tuple (width, height).
    origin:
        The location of the pixel origin. Supported values: ``center``, ``lower left``
    n_components:
        Number of components in input column.
    pixel_column:
        Name of the input column with pixel data.
    output_column:
        Name of the output column with centered pixel data.
    """
    if output_column is None:
        output_column = pixel_column

    if origin == 'center':
        origin_offset = (0.0, 0.0)
    elif origin == 'lower left':
        origin_offset = ((screen_resolution[0] - 1) / 2, (screen_resolution[1] - 1) / 2)
    else:
        supported_origins = ['center', 'lower left']
        raise ValueError(
            f'value `{origin}` for argument `origin` is invalid. '
            f' Valid values are: {supported_origins}',
        )

    centered_pixels = pl.concat_list(
        [
            pl.col(pixel_column).list.get(component) - origin_offset[component % 2]
            for component in range(n_components)
        ],
    ).alias(output_column)
    return centered_pixels
