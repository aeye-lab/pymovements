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

from functools import partial
from typing import Any

import numpy as np
import polars as pl

from pymovements.gaze.transforms_pl.center_origin import center_origin
from pymovements.gaze.transforms_pl.transforms_library import register_transform
from pymovements.utils import checks


def helper(s: Any, distance_px: float) -> pl.Series:
    """This function is a workaround to get complete coverage by testing this function
    explicitly."""
    return pl.Series(np.arctan2(s[0], distance_px))


@register_transform
def pix2deg(
        *,
        screen_px: int,
        screen_cm: float,
        distance_cm: float,
        origin: str,
        pixel_column: str,
        position_column: str,
) -> pl.Expr:
    """Converts pixel screen coordinates to degrees of visual angle.

    Parameters
    ----------
    screen_px:
        Size of screen in pixels.
    screen_cm:
        Size of screen in centimeters.
    distance_cm:
        Eye-to-screen distance in centimeters
    origin:
        The location of the pixel origin. Supported values: ``center``, ``lower left``. See also
        py:func:`~pymovements.gaze.transform.center_origin` for more information.
    pixel_column:
        The input pixel column name.
    position_column:
        The output position column name.
    """
    _check_screen_scalar(screen_px=screen_px, screen_cm=screen_cm, distance_cm=distance_cm)

    centered_pixels = center_origin(screen_px=screen_px, origin=origin, pixel_column=pixel_column)

    # Compute eye-to-screen-distance in pixel units.
    distance_px = distance_cm * (screen_px / screen_cm)

    # Compute positions as radians using arctan2.
    radians = pl.map([centered_pixels], partial(helper, distance_px=distance_px))

    # 180 / pi transforms radians to degrees.
    degrees = radians * (180 / np.pi)

    return degrees.alias(position_column)


def _check_screen_scalar(**kwargs: Any) -> None:
    """Check if all screen values are scalars and are greather than zero."""
    checks.check_is_scalar(**kwargs)
    checks.check_is_greater_than_zero(**kwargs)
