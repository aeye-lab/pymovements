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
"""
Transforms module.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import polars as pl
import scipy


def center_origin(
        *,
        screen_px: int,
        origin: str,
        pixel_column: str = 'pixel',
) -> pl.Expr:

    if origin == 'center':
        centered_pixels = pl.col(pixel_column)

    elif origin == 'lower left':
        centered_pixels = pl.col(pixel_column) - ((screen_px - 1) / 2)

    else:
        supported_origins = ['center', 'lower left']
        raise ValueError(
            f'value `{origin}` for argument `origin` is invalid. '
            f' Valid values are: {supported_origins}',
        )

    # If the sequence is empty, just forward sequence.
    return pl.when(pl.all().len() == 0).then(pl.all()).otherwise(centered_pixels)


def pix2deg(
        *,
        screen_px: int,
        screen_cm: float,
        distance_cm: float,
        origin: str,
        pixel_column: str = 'pixel',
        position_column: str = 'position',
) -> pl.Expr:
    _check_screen_scalar(screen_px=screen_px, screen_cm=screen_cm, distance_cm=distance_cm)

    centered_pixels = center_origin(screen_px=screen_px, origin=origin, pixel_column=pixel_column)

    # Compute eye-to-screen-distance in pixel units.
    distance_px = distance_cm * (screen_px / screen_cm)

    # Compute positions as radians using arctan2.
    radians = pl.map([centered_pixels], lambda s: np.arctan2(s[0], distance_px))

    # 180 / pi transforms radians to degrees.
    degrees = radians * (180 / np.pi)

    return degrees.alias(position_column)


def savitzky_golay(
        *,
        window_length: int,
        degree: int,
        derivative: int = 0,
        sampling_rate: float = 1.0,
        padding: str | float | int | None = 'nearest',
) -> pl.Expr:
    """Apply a 1-D Savitzky-Golay filter to a column. :cite:p:`SavitzkyGolay1964`

    Parameters
    ----------
    window_length : int
        The length of the filter window (i.e., the number of coefficients).
        If `mode` is 'interp', `window_length` must be less than or equal
        to the size of `x`.
    degree : int
        The degree of the polynomial used to fit the samples.
        `degree` must be less than `window_length`.
    derivative : int, optional
        The order of the derivative to compute. This must be a
        nonnegative integer. The default is 0, which means to filter
        the data without differentiating.
    sampling_rate : float, optional
        The spacing of the samples to which the filter will be applied.
        This is only used if deriv > 0. Default is 1.0.
    padding : str or float, optional
        Must be 'mirror', 'constant', 'nearest', 'wrap' or 'interp'. This
        determines the type of extension to use for the padded signal to
        which the filter is applied.  When `mode` is 'constant', the padding
        value is given by `cval`.  See the Notes for more details on 'mirror',
        'constant', 'wrap', and 'nearest'.
        When the 'interp' mode is selected (the default), no extension
        is used.  Instead, a degree `polyorder` polynomial is fit to the
        last `window_length` values of the edges, and this polynomial is
        used to evaluate the last `window_length // 2` output values.

    Returns
    -------
    polars.Expr
        The respective polars expression

    -----
    Details on the `padding` options:
        'mirror':
            Repeats the values at the edges in reverse order. The value
            closest to the edge is not included.
        'nearest':
            The extension contains the nearest input value.
        'wrap':
            The extension contains the values from the other end of the array.
        'constant':
            The extension contains the value given by the `cval` argument.
    For example, if the input is [1, 2, 3, 4, 5, 6, 7, 8], and
    `window_length` is 7, the following shows the extended data for
    the various `mode` options (assuming `cval` is 0)::
        mode       |   Ext   |         Input          |   Ext
        -----------+---------+------------------------+---------
        None       | -  -  - | 1  2  3  4  5  6  7  8 | -  -  -
        0          | 0  0  0 | 1  2  3  4  5  6  7  8 | 0  0  0
        1          | 1  1  1 | 1  2  3  4  5  6  7  8 | 1  1  1
        'nearest'  | 1  1  1 | 1  2  3  4  5  6  7  8 | 8  8  8
        'mirror'   | 4  3  2 | 1  2  3  4  5  6  7  8 | 7  6  5
        'wrap'     | 6  7  8 | 1  2  3  4  5  6  7  8 | 1  2  3

    """
    _check_window_length(window_length=window_length)
    _check_degree(degree=degree, window_length=window_length)
    _check_derivative(derivative=derivative)
    _check_padding(padding=padding)

    delta = 1 / sampling_rate

    constant_value = 0.0
    if isinstance(padding, (int, float)):
        constant_value = padding
        padding = 'constant'
    elif padding is None:
        padding = 'interp'

    # If the sequence is empty, don't use apply but forward sequence.
    return pl.when(pl.all().len() == 0).then(pl.all()).otherwise(
        pl.apply(
            '*',
            lambda s: scipy.signal.savgol_filter(
                x=s[0],
                window_length=window_length,
                polyorder=degree,
                deriv=derivative,
                delta=delta,
                axis=0,
                mode=padding,
                cval=constant_value,
            ),
        ).arr.explode(),  # Use explode to transform array to pl.Series
    )


def pos2vel(
        *,
        sampling_rate: float,
        method: str,
        window_length: int | None = None,
        degree: int | None = None,
        padding: str | float | int | None = 'nearest',
) -> pl.Expr:

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


def pos2acc(
        *,
        sampling_rate: float,
        window_length: int,
        degree: int,
        padding: str | float | int | None = 'nearest',
) -> pl.Expr:
    return savitzky_golay(
        window_length=window_length,
        degree=degree,
        sampling_rate=sampling_rate,
        padding=padding,
        derivative=2,
    )


def downsample(
        factor: int,
        *,
        column: str | Iterable[str] = '*',
) -> pl.Expr:
    _check_downsample_factor(factor)

    return pl.col(column).take_every(n=factor)


def norm(
        columns: tuple[str, str],
) -> pl.Expr:
    x = pl.col(columns[0])
    y = pl.col(columns[1])
    return (x.pow(2) + y.pow(2)).sqrt()


def _check_screen_scalar(**kwargs: Any) -> None:
    for key, value in kwargs.items():

        if not isinstance(value, (int, float)):
            raise TypeError(
                f'`{key}` must be of type int or float but is of type {type(value)}',
            )

        if value == 0:
            raise ValueError(f'`{key}` must not be zero')

        if value < 0:
            raise ValueError(f'`{key}` must not be negative, but is {value}')


def _check_window_length(window_length: Any) -> None:
    _check_is_int(window_length=window_length)
    _check_is_greater_than_zero(degree=window_length)


def _check_degree(degree: Any, window_length: int) -> None:
    _check_is_int(degree=degree)
    _check_is_greater_than_zero(degree=degree)

    if degree >= window_length:
        raise ValueError("'degree' must be less than 'window_length'")


def _check_padding(padding: Any) -> None:
    if not isinstance(padding, (float, int, str)) and padding is not None:
        raise TypeError(
            f"'padding' must be of type 'str', 'int', 'float' or None"
            f"' but is of type '{type(padding).__name__}'",
        )

    if isinstance(padding, str):
        supported_padding_modes = ['nearest', 'mirror', 'wrap']
        if padding not in supported_padding_modes:
            raise ValueError(
                f"Invalid 'padding' value '{padding}'."
                'Choose a valid padding string, a scalar, or None.'
                f' Valid padding strings are: {supported_padding_modes}',
            )


def _check_derivative(derivative: Any) -> None:
    _check_is_int(derivative=derivative)
    _check_is_positive_value(derivative=derivative)


def _check_downsample_factor(factor: Any) -> None:
    _check_is_int(factor=factor)
    _check_is_positive_value(factor=factor)


def _check_is_int(**kwargs: Any) -> None:
    for key, value in kwargs.items():
        if not isinstance(value, int):
            raise TypeError(
                f"'{key}' must be of type 'int' but is of type '{type(value).__name__}'",
            )


def _check_is_greater_than_zero(**kwargs: float | int) -> None:
    for key, value in kwargs.items():
        if value < 1:
            raise ValueError(f"'{key}' must be greater than zero but is {value}")


def _check_is_positive_value(**kwargs: float | int) -> None:
    for key, value in kwargs.items():
        if value < 0:
            raise ValueError(f"'{key}' must not be negative but is {value}")
