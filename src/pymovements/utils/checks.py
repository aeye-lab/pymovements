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
This module holds basic checks which will be reused in other modules.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def check_no_zeros(variable: Any, name: str = 'variable') -> None:
    """
    Check if variable, or if it is iterable, any of its components are zero.
    """
    # construct error message first
    error_message = f'{name} must not be zero'

    # ducktyping check if variable is iterable
    try:
        _ = iter(variable)

    # variable is not iterable, simple check for zero
    except TypeError as exception:
        if variable == 0:
            raise ValueError(error_message) from exception

    # variable is iterable, check each element for zero
    else:
        error_message = 'each component in ' + error_message

        for variable_component in variable:
            if variable_component == 0:
                raise ValueError(error_message)


def check_nan_both_channels(arr: np.ndarray) -> None:
    """
    Checks if all nans occur at the same time steps for both channels.
    """
    # sanity check: horizontal and vertical gaze coordinates missing
    # values at the same time (Eyelink eyetracker never records only
    # one coordinate)
    if not np.array_equal(np.isnan(arr[:, 0]), np.isnan(arr[:, 1])):
        raise ValueError(
            'nans must occur at the same steps of horizontal and vertical direction',
        )


def check_shapes_positions_velocities(positions: np.ndarray, velocities: np.ndarray) -> None:
    """Checks if positions and velocities are of shape ``(N, 2)`` and shape is equal for both.

    Parameters
    ----------
    positions : np.ndarray
        The positions array.
    velocities : np.ndarray
        The velocities array.

    Raises
    ------
    ValueError
        If positions or velocities are not of shape ``(N, 2)`` or the shape is not equal for both.
    """
    # make sure positions and velocities have shape (N, 2)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError(f'positions must have shape (N, 2) but have shape {positions.shape}')
    if velocities.ndim != 2 or velocities.shape[1] != 2:
        raise ValueError(f'velocities must have shape (N, 2) but have shape {velocities.shape}')

    # Check matching shape for positions and velocities
    if positions.shape != velocities.shape:
        raise ValueError(
            f'shape of positions {positions.shape} does not match'
            f' shape of velocities {velocities.shape}',
        )


def check_two_kwargs(**kwargs) -> None:
    """Check if exactly two keyword arguments are given.

    Parameters
    ----------
    kwargs
        Keyword argument dictionary.

    Raises
    ------
    ValueError
        If number of keyword arguments is not 2.
    """
    if len(kwargs) != 2:
        raise ValueError('there must be exactly two keyword arguments in kwargs')


def check_is_mutual_exclusive(**kwargs) -> None:
    """Check if at most one of two values is not None.

    Parameters
    ----------
    kwargs
        Keyword argument dictionary with 2 keyword arguments.

    Raises
    ------
    ValueError
        If more than one value is not None, or if number of keyword arguments is not 2.

    """
    check_two_kwargs(**kwargs)

    key_1, key_2 = (key for _, key in zip(range(2), kwargs.keys()))
    value_1 = kwargs[key_1]
    value_2 = kwargs[key_2]

    if (value_1 is not None) and (value_2 is not None):
        raise ValueError(
            f'The arguments "{key_1}" and "{key_2}" are mutually exclusive.',
        )


def check_is_none_is_mutual(**kwargs) -> None:
    """Check if two values are either both None or both have a value.

    Parameters
    ----------
    kwargs
        Keyword argument dictionary with 2 keyword arguments.

    Raises
    ------
    ValueError
        If exclusively one of the keyword argument values is None, or if number of keyword arguments
        is not 2.

    """
    check_two_kwargs(**kwargs)

    key_1, key_2 = (key for _, key in zip(range(2), kwargs.keys()))
    value_1 = kwargs[key_1]
    value_2 = kwargs[key_2]

    if not (value_1 is None) == (value_2 is None):
        raise ValueError(
            f'The arguments "{key_1}" and "{key_2}" must be either both None or both not None.',
        )


def check_is_length_matching(**kwargs) -> None:
    """Check if two sequences are of equal length.

    Parameters
    ----------
    kwargs
        Keyword argument dictionary with 2 keyword arguments. Both values must be sequences.

    Raises
    ------
    ValueError
        If both sequences are of equal length , or if number of keyword arguments is not 2.
    """
    check_two_kwargs(**kwargs)

    key_1, key_2 = (key for _, key in zip(range(2), kwargs.keys()))
    value_1 = kwargs[key_1]
    value_2 = kwargs[key_2]

    if not len(value_1) == len(value_2):
        raise ValueError(f'The sequences "{key_1}" and "{key_2}" must be of equal length.')
