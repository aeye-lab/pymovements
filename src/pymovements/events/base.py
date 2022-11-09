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
