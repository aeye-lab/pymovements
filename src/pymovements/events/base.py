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


class Event:
    """
    Base Event class.

    Parameters
    ----------
    name: str
        Name of event.
    onset: int
        Starting index of event (included).
    offset: int
        Ending index of event (excluded).

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
        """
        return self.offset - self.onset


class Fixation(Event):
    """
    Fixation class.

    Parameters
    ----------
    onset: int
        Starting index of event (included).
    offset: int
        Ending index of event (excluded).
    position: tuple[float, float]
        (x, y) position of fixation

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
        super().__init__(name=self._name, onset=onset, offset=offset)
        self.position = position


class Saccade(Event):
    """
    Saccade class.

    Parameters
    ----------
    onset: int
        Starting index of event (included).
    offset: int
        Ending index of event (excluded).

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
        super().__init__(name=self._name, onset=onset, offset=offset)
