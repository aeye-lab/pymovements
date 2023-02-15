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

from pymovements.utils.decorators import auto_str


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
    """

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
    """
    _name = 'fixation'

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
