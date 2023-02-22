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
This module holds all the main Event classes used for event detection.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from typing_extensions import Protocol

from pymovements.utils.decorators import auto_str


class EventDetectionCallable(Protocol):
    """Minimal interface to be implemented by all event detection methods."""

    def __call__(
            self,
            positions: list[list[float]] | list[tuple[float, float]] | np.ndarray,
            velocities: list[list[float]] | list[tuple[float, float]] | np.ndarray,
            **kwargs,
    ) -> pl.DataFrame:
        """Minimal interface to be implemented by all event detection methods.

        Parameters
        ----------
        positions: array-like, shape (N, 2)
            Continuous 2D position time series
        velocities: array-like, shape (N, 2)
            Corresponding continuous 2D velocity time series.
        **kwargs:
            Additional keyword arguments for the specific event detection method.

        Returns
        -------
        pl.DataFrame
            A dataframe with detected events as rows.
        """


@auto_str
class Event:
    """
    Base Event class.

    Attributes
    ----------
    name: str
        Name of event.
    onset: int
        Starting index of event (included).
    offset: int
        Ending index of event (excluded).
    schema: polars.SchemaDict
        Schema for event DataFrame.
    """
    schema: pl.datatypes.SchemaDict = {'type': pl.Utf8, 'onset': pl.Int64, 'offset': pl.Int64}

    def __init__(self, name: str, onset: int, offset: int):
        """
        Parameters
        ----------
        name: str
            Name of event.
        onset: int
            Starting index of event (included).
        offset: int
            Ending index of event (excluded).

        Examples
        --------
        >>> event = Event(
        ...    name="custom_event",
        ...    onset=5,
        ...    offset=10,
        ... )
        >>> print(event)
        Event(name=custom_event, onset=5, offset=10)
        """
        self.name = name
        self.onset = onset
        self.offset = offset

    @property
    def duration(self) -> int:
        """
        Get sample duration of event.

        Returns
        -------
        int
            duration in samples.

        Examples
        --------
        >>> event = Event(
        ...    name="custom_event",
        ...    onset=5,
        ...    offset=10,
        ... )
        >>> event.duration
        5
        """
        return self.offset - self.onset


@auto_str
class Fixation(Event):
    """
    Fixation class.

    Attributes
    ----------
    name: str
        Name of event.
    onset: int
        Starting index of event (included).
    offset: int
        Ending index of event (excluded).
    position: tuple[float, float]
        (x, y) position of fixation
    schema: polars.SchemaDict
        Schema for event DataFrame.
    """
    _name = 'fixation'

    schema: pl.datatypes.SchemaDict = {**Event.schema, 'position': pl.List(pl.Float64)}

    def __init__(self, onset: int, offset: int, position: tuple[float, float]):
        """
        Parameters
        ----------
        onset: int
            Starting index of event (included).
        offset: int
            Ending index of event (excluded).
        position: tuple[float, float]
            (x, y) position of fixation

        Examples
        --------
        >>> fixation = Fixation(
        ...    onset=5,
        ...    offset=10,
        ...    position=(125.1, 852.3),
        ... )
        >>> print(fixation)
        Fixation(name=fixation, onset=5, offset=10, position=(125.1, 852.3))
        """
        super().__init__(name=self._name, onset=onset, offset=offset)
        self.position = position


@auto_str
class Saccade(Event):
    """
    Saccade class.

    Attributes
    ----------
    name: str
        Name of event.
    onset: int
        Starting index of event (included).
    offset: int
        Ending index of event (excluded).
    schema: polars.SchemaDict
        Schema for event DataFrame.
    """
    _name = 'saccade'

    def __init__(self, onset: int, offset: int):
        """
        Parameters
        ----------
        onset: int
            Starting index of event (included).
        offset: int
            Ending index of event (excluded).

        Examples
        --------
        >>> saccade = Saccade(
        ...    onset=8,
        ...    offset=10,
        ... )
        >>> print(saccade)
        Saccade(name=saccade, onset=8, offset=10)
        """

        super().__init__(name=self._name, onset=onset, offset=offset)
