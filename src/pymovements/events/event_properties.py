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
def amplitude(
        *,
        position_column: str = 'position',
        n_components: int = 2,
) -> pl.Expr:
    """Amplitude of an event.

    Parameters
    ----------
    position_column
        The column name of the position tuples.
    n_components:
        Number of positional components. Usually these are the two components yaw and pitch.

    Raises
    ------
    ValueError
        If number of components is not 2.
    """
    _check_has_two_componenents(n_components)

    x_position = pl.col(position_column).arr.get(0)
    y_position = pl.col(position_column).arr.get(1)

    return (
        (x_position.max() - x_position.min()).pow(2)
        + (y_position.max() - y_position.min()).pow(2)
    ).sqrt()


@register_event_property
def dispersion(
        *,
        position_column: str = 'position',
        n_components: int = 2,
) -> pl.Expr:
    """Dispersion of an event.

    Parameters
    ----------
    position_column
        The column name of the position tuples.
    n_components:
        Number of positional components. Usually these are the two components yaw and pitch.

    Raises
    ------
    ValueError
        If number of components is not 2.
    """
    _check_has_two_componenents(n_components)

    x_position = pl.col(position_column).arr.get(0)
    y_position = pl.col(position_column).arr.get(1)

    return x_position.max() - x_position.min() + y_position.max() - y_position.min()


@register_event_property
def disposition(
        *,
        position_column: str = 'position',
        n_components: int = 2,
) -> pl.Expr:
    """Disposition of an event.

    Parameters
    ----------
    position_column
        The column name of the position tuples.
    n_components:
        Number of positional components. Usually these are the two components yaw and pitch.

    Raises
    ------
    TypeError
        If position_columns not of type tuple, position_columns not of length 2, or elements of
        position_columns not of type str.
    """
    _check_has_two_componenents(n_components)

    x_position = pl.col(position_column).arr.get(0)
    y_position = pl.col(position_column).arr.get(1)

    return (
        (x_position.head(n=1) - x_position.reverse().head(n=1)).pow(2)
        + (y_position.head(n=1) - y_position.reverse().head(n=1)).pow(2)
    ).sqrt()


@register_event_property
def duration() -> pl.Expr:
    """Duration of an event.

    The duration is defined as the difference between offset time and onset time.
    """
    return pl.col('offset') - pl.col('onset')


@register_event_property
def location(
        method: str = 'mean',
        *,
        position_column: str = 'position',
        n_components: int = 2,
) -> pl.Expr:
    """Location of an event.

    Parameters
    ----------
    method
        The centroid method to be used for calculation. Supported methods are ``mean``, ``median``.
        Defaults to 'mean'.
    position_column
        The column name of the position tuples.
    n_components:
        Number of positional components. Usually these are the two components yaw and pitch.

    Raises
    ------
    ValueError
        If method is not one of the supported methods.
    """
    if method not in ['mean', 'median']:
        raise ValueError(
            f"Method '{method}' not supported. "
            f"Please choose one of the following: ['mean', 'median'].",
        )

    component_expressions = []
    for component in range(n_components):
        position_component = (
            pl.col(position_column)
            .arr.slice(0, None)
            .arr.get(component)
        )

        if method == 'mean':
            expression_component = position_component.mean()

        if method == 'median':
            expression_component = position_component.median()

        component_expressions.append(expression_component)

    # Not sure why first() is needed here, but an outer list is being created somehow.
    return pl.concat_list(component_expressions).first()


@register_event_property
def peak_velocity(
        *,
        velocity_column: str = 'velocity',
        n_components: int = 2,
) -> pl.Expr:
    """Peak velocity of an event.

    Parameters
    ----------
    velocity_column
        The column name of the velocity tuples.
    n_components:
        Number of positional components. Usually these are the two components yaw and pitch.

    Raises
    ------
    ValueError
        If number of components is not 2.
    """
    _check_has_two_componenents(n_components)

    x_velocity = pl.col(velocity_column).arr.get(0)
    y_velocity = pl.col(velocity_column).arr.get(1)

    return (x_velocity.pow(2) + y_velocity.pow(2)).sqrt().max()


def _check_has_two_componenents(n_components: int) -> None:
    """Check that number of componenents is two."""
    if n_components != 2:
        raise ValueError('data must have exactly two components')
