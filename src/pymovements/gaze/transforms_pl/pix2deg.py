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
"""Module for py:func:`pymovements.gaze.transforms.pix2deg`"""
from __future__ import annotations

import numpy as np
import polars as pl

from pymovements.gaze.transforms_pl.center_origin import center_origin
from pymovements.gaze.transforms_pl.transforms_library import register_transform
from pymovements.utils import checks


@register_transform
def pix2deg(
        *,
        screen_resolution: tuple[int, int],
        screen_size: tuple[float, float],
        distance: float,
        origin: str,
        n_components: int,
        pixel_column: str = 'pixel',
        position_column: str = 'position',
) -> pl.Expr:
    """Converts pixel screen coordinates to degrees of visual angle.

    Parameters
    ----------
    screen_resolution:
        Pixel screen resolution as tuple (width, height).
    screen_size:
        Screen size in centimeters as tuple (width, height).
    distance:
        Eye-to-screen distance in centimeters
    origin:
        The location of the pixel origin. Supported values: ``center``, ``lower left``. See also
        py:func:`~pymovements.gaze.transform.center_origin` for more information.
    n_components:
        Number of components in input column.
    pixel_column:
        The input pixel column name.
    position_column:
        The output position column name.
    """
    _check_screen_resolution(screen_resolution)
    _check_screen_size(screen_size)
    _check_distance(distance)

    centered_pixels = center_origin(
        screen_resolution=screen_resolution,
        origin=origin,
        n_components=n_components,
        pixel_column=pixel_column,
    )

    # Compute eye-to-screen-distance in pixel units.
    distance_pixels = tuple(
        distance * (screen_px / screen_cm)
        for screen_px, screen_cm in zip(screen_resolution, screen_size)
    )

    degree_components = [
        centered_pixels.list.get(component).map(
            lambda s: np.arctan2(
                s, distance_pixels[component % 2],  # pylint: disable=cell-var-from-loop
            ),
        ) * (180 / np.pi)
        for component in range(n_components)
    ]

    return pl.concat_list(degree_components).alias(position_column)


def _check_distance(distance: float) -> None:
    """Check if all screen values are scalars and are greather than zero."""
    checks.check_is_scalar(distance=distance)
    checks.check_is_greater_than_zero(distance=distance)


def _check_screen_resolution(screen_resolution: tuple[int, int]) -> None:
    """Check screen resolution value."""
    if screen_resolution is None:
        raise TypeError('screen_resolution must not be None')

    if not isinstance(screen_resolution, (tuple, list)):
        raise TypeError(
            'screen_resolution must be of type tuple[int, int],'
            f' but is of type {type(screen_resolution).__name__}',
        )

    if len(screen_resolution) != 2:
        raise ValueError(
            f'screen_resolution must have length of 2, but is of length {len(screen_resolution)}',
        )

    for element in screen_resolution:
        checks.check_is_scalar(screen_resolution=element)
        checks.check_is_greater_than_zero(screen_resolution=element)


def _check_screen_size(screen_size: tuple[float, float]) -> None:
    """Check screen size value."""
    if screen_size is None:
        raise TypeError('screen_size must not be None')

    if not isinstance(screen_size, (tuple, list)):
        raise TypeError(
            'screen_size must be of type tuple[int, int],'
            f' but is of type {type(screen_size).__name__}',
        )

    if len(screen_size) != 2:
        raise ValueError(f'screen_size must have length of 2, but is of length {len(screen_size)}')

    for element in screen_size:
        checks.check_is_scalar(screen_size=element)
        checks.check_is_greater_than_zero(screen_size=element)
