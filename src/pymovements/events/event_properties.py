# Copyright (c) 2023 The pymovements Project Authors
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
"""This module holds all supported event properties."""
from __future__ import annotations

from collections.abc import Callable

import polars as pl


EVENT_PROPERTIES: dict[str, Callable] = {}


def register_event_property(function: Callable) -> Callable:
    """Register a function as a valid property."""
    EVENT_PROPERTIES[function.__name__] = function
    return function


@register_event_property
def duration() -> pl.Expr:
    """Duration of an event.

    The duration is defined as the difference between offset time and onset time.
    """
    return pl.col('offset') - pl.col('onset')


@register_event_property
def peak_velocity(velocity_columns: tuple[str, str] = ('x_vel', 'y_vel')) -> pl.Expr:
    """Peak velocity of an event.

    Parameters
    ----------
    velocity_columns
        The column names of the pitch and yaw velocity components.

    Raises
    ------
    TypeError
        If velocity_columns not of type tuple, velocity_columns not of length 2, or elements of
        velocity_columns not of type str.
    """
    _check_velocity_columns(velocity_columns)

    x_velocity = pl.col(velocity_columns[0])
    y_velocity = pl.col(velocity_columns[1])

    return (x_velocity.pow(2) + y_velocity.pow(2)).sqrt().max()


@register_event_property
def dispersion(position_columns: tuple[str, str] = ('x_pos', 'y_pos')) -> pl.Expr:
    """Dispersion of an event.

    Parameters
    ----------
    position_columns
        The column names of the pitch and yaw position components.

    Raises
    ------
    TypeError
        If position_columns not of type tuple, position_columns not of length 2, or elements of
        position_columns not of type str.
    """
    _check_position_columns(position_columns)

    x_position = pl.col(position_columns[0])
    y_position = pl.col(position_columns[1])

    return x_position.max() - x_position.min() + y_position.max() - y_position.min()


@register_event_property
def amplitude(position_columns: tuple[str, str] = ('x_pos', 'y_pos')) -> pl.Expr:
    """Amplitude of an event.

    Parameters
    ----------
    position_columns
        The column names of the pitch and yaw position components.

    Raises
    ------
    TypeError
        If position_columns not of type tuple, position_columns not of length 2, or elements of
        position_columns not of type str.
    """
    _check_position_columns(position_columns)

    x_position = pl.col(position_columns[0])
    y_position = pl.col(position_columns[1])

    return (
        (x_position.max() - x_position.min()).pow(2)
        + (y_position.max() - y_position.min()).pow(2)
    ).sqrt()


@register_event_property
def disposition(position_columns: tuple[str, str] = ('x_pos', 'y_pos')) -> pl.Expr:
    """Disposition of an event.

    Parameters
    ----------
    position_columns
        The column names of the pitch and yaw position components.

    Raises
    ------
    TypeError
        If position_columns not of type tuple, position_columns not of length 2, or elements of
        position_columns not of type str.
    """
    _check_position_columns(position_columns)

    x_position = pl.col(position_columns[0])
    y_position = pl.col(position_columns[1])

    return (
        (x_position.head(n=1) - x_position.reverse().head(n=1)).pow(2)
        + (y_position.head(n=1) - y_position.reverse().head(n=1)).pow(2)
    ).sqrt()


def _check_position_columns(position_columns: tuple[str, str]) -> None:
    """Check if position_columns is of type tuple[str, str]."""
    if not isinstance(position_columns, tuple):
        raise TypeError(
            'position_columns must be of type tuple[str, str]'
            f' but is of type {type(position_columns).__name__}',
        )
    if len(position_columns) != 2:
        raise TypeError(
            f'position_columns must be of length of 2 but is of length {len(position_columns)}',
        )
    if not all(isinstance(velocity_column, str) for velocity_column in position_columns):
        raise TypeError(
            'position_columns must be of type tuple[str, str] but is '
            f'tuple[{type(position_columns[0]).__name__}, {type(position_columns[1]).__name__}]',
        )


def _check_velocity_columns(velocity_columns: tuple[str, str]) -> None:
    """Check if velocity_columns is of type tuple[str, str]."""
    if not isinstance(velocity_columns, tuple):
        raise TypeError(
            'velocity_columns must be of type tuple[str, str]'
            f' but is of type {type(velocity_columns).__name__}',
        )
    if len(velocity_columns) != 2:
        raise TypeError(
            f'velocity_columns must be of length of 2 but is of length {len(velocity_columns)}',
        )
    if not all(isinstance(velocity_column, str) for velocity_column in velocity_columns):
        raise TypeError(
            'velocity_columns must be of type tuple[str, str] but is '
            f'tuple[{type(velocity_columns[0]).__name__}, {type(velocity_columns[1]).__name__}]',
        )
