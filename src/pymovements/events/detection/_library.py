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
"""This module holds the EventDetectionLibrary class."""
from __future__ import annotations

from collections.abc import Callable

from pymovements.events.frame import EventDataFrame


class EventDetectionLibrary:
    """Provides access by name to event detection methods.

    Attributes
    ----------
    `methods`:
        Dictionary of event detection methods.
    """

    methods: dict[str, Callable[..., EventDataFrame]] = {}

    @classmethod
    def add(cls, method: Callable[..., EventDataFrame]) -> None:
        """Add an event detection method to the library.

        Parameter
        ---------
        method
            The event detection method to add to the library.
        """
        cls.methods[method.__name__] = method

    @classmethod
    def get(cls, name: str) -> Callable[..., EventDataFrame]:
        """Get event detection method py name.

        Parameter
        ---------
        name
            Name of the event detection method in the library.
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
            True if EventDetectionLibrary contains method with given name, else False.
        """
        return name in cls.methods


def register_event_detection(
        method: Callable[..., EventDataFrame],
) -> Callable[..., EventDataFrame]:
    """Register an event detection method."""
    EventDetectionLibrary.add(method)
    return method
