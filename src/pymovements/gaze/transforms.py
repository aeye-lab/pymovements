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
"""Module for py:func:`pymovements.gaze.transforms."""
from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any
from typing import TypeVar

import numpy as np
import polars as pl
import scipy

from pymovements.utils import checks

TransformMethod = TypeVar('TransformMethod', bound=Callable[..., pl.Expr])


class TransformLibrary:
    """Provides access by name to transformation methods.

    Attributes
    ----------
    `methods`:
        Dictionary of transformation methods.
    """

    methods: dict[str, Callable[..., pl.Expr]] = {}

    @classmethod
    def add(cls, method: Callable[..., pl.Expr]) -> None:
        """Add a transformation method to the library.

        Parameter
        ---------
        method
            The transformation method to add to the library.
        """
        cls.methods[method.__name__] = method

    @classmethod
    def get(cls, name: str) -> Callable[..., pl.Expr]:
        """Get transformation method py name.

        Parameter
        ---------
        name
            Name of the transformation method in the library.
        """
        return cls.methods[name]

    @classmethod
    def __contains__(cls, name: str) -> bool:
        """Check if class contains method of given name.

        Parameters
        ----------
        name: str
            Name of the method to check.

        Returns
        -------
        bool
            True if TransformsLibrary contains method with given name, else False.
        """
        return name in cls.methods


def register_transform(method: TransformMethod) -> TransformMethod:
    """Register a transform method."""
    TransformLibrary.add(method)
    return method


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


@register_transform
def downsample(
        *,
        factor: int,
) -> pl.Expr:
    """Downsample gaze data by an integer factor.

    Downsampling is done by taking every `nth` sample specified by the downsampling factor.

    Parameters
    ----------
    factor:
        Downsample factor.
    """
    checks.check_is_int(factor=factor)
    checks.check_is_positive_value(factor=factor)

    return pl.all().take_every(n=factor)


@register_transform
def norm(
        *,
        columns: tuple[str, str],
) -> pl.Expr:
    r"""Take the norm of a 2D series.

    The norm is defined by :math:`\sqrt{x^2 + y^2}` with :math:`x` being the yaw component and
    :math:`y` being the pitch component of a coordinate.

    Parameters
    ----------
    columns:
        Columns to take norm of.
    """
    x = pl.col(columns[0])
    y = pl.col(columns[1])
    return (x.pow(2) + y.pow(2)).sqrt()


@register_transform
def pix2deg(
        *,
        screen_resolution: tuple[int, int],
        screen_size: tuple[float, float],
        distance: float | str,
        origin: str,
        n_components: int,
        pixel_column: str = 'pixel',
        position_column: str = 'position',
) -> pl.Expr:
    """Convert pixel screen coordinates to degrees of visual angle.

    Parameters
    ----------
    screen_resolution:
        Pixel screen resolution as tuple (width, height).
    screen_size:
        Screen size in centimeters as tuple (width, height).
    distance:
        Must be either a scalar or a string. If a scalar is passed, it is interpreted as the
        Eye-to-screen distance in centimeters. If a string is passed, it is interpreted as the name
        of a column containing the Eye-to-screen distance in millimiters for each sample.
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

    centered_pixels = center_origin(
        screen_resolution=screen_resolution,
        origin=origin,
        n_components=n_components,
        pixel_column=pixel_column,
    )

    if isinstance(distance, (float, int)):
        _check_distance(distance)
        distance_series = pl.lit(distance)
    elif isinstance(distance, str):
        # True division by 10 is needed to convert distance from mm to cm
        distance_series = pl.col(distance).truediv(10)
    else:
        raise TypeError(
            f'`distance` must be of type `float`, `int` or `str`, but is of type'
            f'`{type(distance).__name__}`',
        )

    distance_pixels = pl.concat_list([
        distance_series.mul(screen_resolution[component % 2] / screen_size[component % 2])
        for component in range(n_components)
    ])

    degree_components = [
        pl.arctan2(
            centered_pixels.list.get(component), distance_pixels.list.get(component),
        ) * (180 / np.pi)
        for component in range(n_components)
    ]

    return pl.concat_list(list(degree_components)).alias(position_column)


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
        The padding method to use. See ``savitzky_golay`` for details.
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


@register_transform
def pos2vel(
        *,
        sampling_rate: float,
        method: str,
        n_components: int,
        degree: int | None = None,
        window_length: int | None = None,
        padding: str | float | int | None = 'nearest',
        position_column: str = 'position',
        velocity_column: str = 'velocity',
) -> pl.Expr:
    """Compute velocitiy data from positional data.

    Parameters
    ----------
    sampling_rate:
        Sampling rate of input time series.
    method:
        The method to use for velocity calculation.
    degree:
        The degree of the polynomial to use. This has only an effect if using ``savitzky_golay`` as
        calculation method.
    window_length:
        The window size to use. This has only an effect if using ``savitzky_golay`` as calculation
        method.
    padding:
        The padding to use.  This has only an effect if using ``savitzky_golay`` as calculation
        method.
    n_components:
        Number of components in input column.
    position_column:
        The input position column name.
    velocity_column:
        The output velocity column name.

    Notes
    -----
    There are three methods available for velocity calculation:

    * ``savitzky_golay``: velocity is calculated by a polynomial of fixed degree and window length.
      See :py:func:`~pymovements.gaze.transforms.savitzky_golay` for further details.
    * ``five_point``: velocity is calculated from the difference of the mean values
      of the subsequent two samples and the preceding two samples
    * ``neighbors``: velocity is calculated from difference of the subsequent
      sample and the preceding sample
    * ``preceding``: velocity is calculated from the difference of the current
      sample to the preceding sample
    """
    if method == 'preceding':
        return pl.concat_list(
            [
                pl.col(position_column).list.get(component)
                .diff(n=1, null_behavior='ignore') * sampling_rate
                for component in range(n_components)
            ],
        ).alias(velocity_column)

    if method == 'neighbors':
        return pl.concat_list(
            [
                (
                    pl.col(position_column).shift(periods=-1).list.get(component)
                    - pl.col(position_column).shift(periods=1).list.get(component)
                ) * (sampling_rate / 2)
                for component in range(n_components)
            ],
        ).alias(velocity_column)

    if method in {'fivepoint', 'smooth'}:
        # Center of window is period 0 and will be filled.
        # mean(arr_-2, arr_-1) and mean(arr_1, arr_2) needs division by two
        # window is now 3 samples long (arr_-1.5, arr_0, arr_1+5)
        # we therefore need a divison by three, all in all it's a division by 6
        return pl.concat_list(
            [
                (
                    pl.col(position_column).shift(periods=-2).list.get(component)
                    + pl.col(position_column).shift(periods=-1).list.get(component)
                    - pl.col(position_column).shift(periods=1).list.get(component)
                    - pl.col(position_column).shift(periods=2).list.get(component)
                ) * (sampling_rate / 6)
                for component in range(n_components)
            ],
        ).alias(velocity_column)

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
            n_components=n_components,
            input_column=position_column,
            output_column=velocity_column,
        )

    supported_methods = ['preceding', 'neighbors', 'fivepoint', 'smooth', 'savitzky_golay']
    raise ValueError(
        f"Unknown method '{method}'. Supported methods are: {supported_methods}",
    )


@register_transform
def savitzky_golay(
        *,
        window_length: int,
        degree: int,
        sampling_rate: float,
        n_components: int,
        input_column: str,
        output_column: str | None = None,
        derivative: int = 0,
        padding: str | float | int | None = 'nearest',
) -> pl.Expr:
    """Apply a 1-D Savitzky-Golay filter to a column|_|:cite:p:`SavitzkyGolay1964`.

    Parameters
    ----------
    sampling_rate : float, optional
        The spacing of the samples to which the filter will be applied.
        This is only used if deriv > 0. Default is 1.0.
    n_components:
        Number of components in input column.
    window_length : int
        The length of the filter window (i.e., the number of coefficients).
        If `padding` is ``None``, `window_length` must be less than or equal
        to the length of the input.
    degree : int
        The degree of the polynomial used to fit the samples.
        `degree` must be less than `window_length`.
    derivative : int, optional
        The order of the derivative to compute. This must be a
        nonnegative integer. The default is 0, which means to filter
        the data without differentiating.
    padding : str or float, optional
        Must be either ``None``, a scalar or one of the strings ``mirror``, ``nearest`` or ``wrap``.
        This determines the type of extension to use for the padded signal to
        which the filter is applied.
        When passing ``None``, no extension padding is used. Instead, a degree `degree` polynomial
        is fit to the last ``window_length`` values of the edges, and this polynomial is used to
        evaluate the last ``window_length // 2`` output values.
        When passing a scalar value, data will be padded using the passed value.
        See the Notes for more details on the padding methods ``mirror``, ``nearest`` or ``wrap``.
    input_column:
        The input column name.
    output_column:
        The output column name.

    Returns
    -------
    polars.Expr
        The respective polars expression

    Notes
    -----
    Details on the `padding` options:

    * ``None``: No padding extension is used.
    * scalar value (int or float): The padding extension contains the specified scalar value.
    * ``mirror``: Repeats the values at the edges in reverse order. The value closest to the edge is
      not included.
    * ``nearest``: The padding extension contains the nearest input value.
    * ``wrap``: The padding extension contains the values from the other end of the array.

    Given the input is ``[1, 2, 3, 4, 5, 6, 7, 8]``, and
    `window_length` is 7, the following table shows the padded data for
    the various ``padding`` options:

    +-------------+-------------+----------------------------+-------------+
    | mode        |   padding   |           input            |   padding   |
    +=============+=============+============================+=============+
    | ``None``    | ``-  -  -`` | ``1  2  3  4  5  6  7  8`` | ``-  -  -`` |
    +-------------+-------------+----------------------------+-------------+
    | ``0``       | ``0  0  0`` | ``1  2  3  4  5  6  7  8`` | ``0  0  0`` |
    +-------------+-------------+----------------------------+-------------+
    | ``1``       | ``1  1  1`` | ``1  2  3  4  5  6  7  8`` | ``1  1  1`` |
    +-------------+-------------+----------------------------+-------------+
    | ``nearest`` | ``1  1  1`` | ``1  2  3  4  5  6  7  8`` | ``8  8  8`` |
    +-------------+-------------+----------------------------+-------------+
    | ``mirror``  | ``4  3  2`` | ``1  2  3  4  5  6  7  8`` | ``7  6  5`` |
    +-------------+-------------+----------------------------+-------------+
    | ``wrap``    | ``6  7  8`` | ``1  2  3  4  5  6  7  8`` | ``1  2  3`` |
    +-------------+-------------+----------------------------+-------------+
    """
    _check_window_length(window_length=window_length)
    _check_degree(degree=degree, window_length=window_length)
    _check_derivative(derivative=derivative)
    _check_padding(padding=padding)

    if output_column is None:
        output_column = input_column

    delta = 1 / sampling_rate

    constant_value = 0.0
    if isinstance(padding, (int, float)):
        constant_value = padding
        padding = 'constant'
    elif padding is None:
        padding = 'interp'

    func = partial(
        scipy.signal.savgol_filter,
        window_length=window_length,
        polyorder=degree,
        deriv=derivative,
        delta=delta,
        axis=0,
        mode=padding,
        cval=constant_value,
    )

    return pl.concat_list(
        [
            pl.col(input_column).list.get(component).map(func).list.explode()
            for component in range(n_components)
        ],
    ).alias(output_column)


@register_transform
def smooth(
        *,
        method: str,
        window_length: int,
        n_components: int,
        degree: int | None = None,
        column: str = 'position',
        padding: str | float | int | None = 'nearest',
) -> pl.Expr:
    """
    Smooth data in a column.

    Parameters
    ----------
    method:
        The method to use for smoothing. See Notes for more details.
    window_length
        For ``moving_average`` this is the window size to calculate the mean of the subsequent
        samples. For ``savitzky_golay`` this is the window size to use for the polynomial fit.
        For ``exponential_moving_average`` this is the span parameter.
    n_components:
        Number of components in input column.
    degree:
        The degree of the polynomial to use. This has only an effect if using ``savitzky_golay`` as
        smoothing method. `degree` must be less than `window_length`.
    column:
        The input column name to which the smoothing is applied.
    padding:
        Must be either ``None``, a scalar or one of the strings ``mirror``, ``nearest`` or ``wrap``.
        This determines the type of extension to use for the padded signal to
        which the filter is applied.
        When passing ``None``, no extension padding is used.
        When passing a scalar value, data will be padded using the passed value.
        See the Notes for more details on the padding methods.

    Returns
    -------
    polars.Expr
        The respective polars expression.

    Notes
    -----
    There following methods are available for smoothing:

    * ``savitzky_golay``: Smooth data by applying a Savitzky-Golay filter.
    See :py:func:`~pymovements.gaze.transforms.savitzky_golay` for further details.
    * ``moving_average``: Smooth data by calculating the mean of the subsequent samples.
    Each smoothed sample is calculated by the mean of the samples in the window around the sample.
    * ``exponential_moving_average``: Smooth data by exponentially weighted moving average.

    Details on the `padding` options:

    * ``None``: No padding extension is used.
    * scalar value (int or float): The padding extension contains the specified scalar value.
    * ``mirror``: Repeats the values at the edges in reverse order. The value closest to the edge is
      not included.
    * ``nearest``: The padding extension contains the nearest input value.
    * ``wrap``: The padding extension contains the values from the other end of the array.

    Given the input is ``[1, 2, 3, 4, 5, 6, 7, 8]``, and
    `window_length` is 7, the following table shows the padded data for
    the various ``padding`` options:

    +-------------+-------------+----------------------------+-------------+
    | mode        |   padding   |           input            |   padding   |
    +=============+=============+============================+=============+
    | ``None``    | ``-  -  -`` | ``1  2  3  4  5  6  7  8`` | ``-  -  -`` |
    +-------------+-------------+----------------------------+-------------+
    | ``0``       | ``0  0  0`` | ``1  2  3  4  5  6  7  8`` | ``0  0  0`` |
    +-------------+-------------+----------------------------+-------------+
    | ``1``       | ``1  1  1`` | ``1  2  3  4  5  6  7  8`` | ``1  1  1`` |
    +-------------+-------------+----------------------------+-------------+
    | ``nearest`` | ``1  1  1`` | ``1  2  3  4  5  6  7  8`` | ``8  8  8`` |
    +-------------+-------------+----------------------------+-------------+
    | ``mirror``  | ``4  3  2`` | ``1  2  3  4  5  6  7  8`` | ``7  6  5`` |
    +-------------+-------------+----------------------------+-------------+
    | ``wrap``    | ``6  7  8`` | ``1  2  3  4  5  6  7  8`` | ``1  2  3`` |
    +-------------+-------------+----------------------------+-------------+

    """
    _check_window_length(window_length=window_length)
    _check_padding(padding=padding)

    if method in {'moving_average', 'exponential_moving_average'}:
        pad_kwargs: dict[str, Any] = {'pad_width': 0}
        pad_func = _identity

        if isinstance(padding, (int, float)):
            pad_kwargs['constant_values'] = padding
            padding = 'constant'
        elif padding == 'nearest':
            # option 'nearest' is called 'edge' for np.pad
            padding = 'edge'
        elif padding == 'mirror':
            # option 'mirror' is called 'reflect' for np.pad
            padding = 'reflect'

        if padding is not None:
            pad_kwargs['mode'] = padding
            pad_kwargs['pad_width'] = np.ceil(window_length / 2).astype(int)

            pad_func = partial(
                np.pad,
                **pad_kwargs,
            )

        if method == 'moving_average':

            return pl.concat_list(
                [
                    pl.col(column).list.get(component).map(pad_func).list.explode()
                    .rolling_mean(window_size=window_length, center=True)
                    .shift(periods=pad_kwargs['pad_width'])
                    .slice(pad_kwargs['pad_width'] * 2)
                    for component in range(n_components)
                ],
            ).alias(column)

        return pl.concat_list(
            [
                pl.col(column).list.get(component).map(pad_func).list.explode()
                .ewm_mean(
                    span=window_length,
                    adjust=False,
                    min_periods=window_length,
                ).shift(periods=pad_kwargs['pad_width'])
                .slice(pad_kwargs['pad_width'] * 2)
                for component in range(n_components)
            ],
        ).alias(column)

    if method == 'savitzky_golay':
        if degree is None:
            raise TypeError("'degree' must not be none for method 'savitzky_golay'")

        return savitzky_golay(
            window_length=window_length,
            degree=degree,
            sampling_rate=1,
            padding=padding,
            derivative=0,
            n_components=n_components,
            input_column=column,
            output_column=None,
        )

    supported_methods = ['moving_average', 'exponential_moving_average', 'savitzky_golay']

    raise ValueError(
        f"Unknown method '{method}'. Supported methods are: {supported_methods}",
    )


def _identity(x: Any) -> Any:
    """Identity function as placeholder for None as padding."""
    return x


def _check_window_length(window_length: Any) -> None:
    """Check that window length is an integer and greater than zero."""
    checks.check_is_not_none(window_length=window_length)
    checks.check_is_int(window_length=window_length)
    checks.check_is_greater_than_zero(degree=window_length)


def _check_degree(degree: Any, window_length: int) -> None:
    """Check that polynomial degree is an integer, greater than zero and less than window_length."""
    checks.check_is_not_none(degree=degree)
    checks.check_is_int(degree=degree)
    checks.check_is_greater_than_zero(degree=degree)

    if degree >= window_length:
        raise ValueError("'degree' must be less than 'window_length'")


def _check_padding(padding: Any) -> None:
    """Check if padding argument is valid."""
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
    """Check that derivative has a positive integer value."""
    checks.check_is_int(derivative=derivative)
    checks.check_is_positive_value(derivative=derivative)
