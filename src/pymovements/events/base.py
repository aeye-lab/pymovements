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
        >>> print(event) # doctest: +NORMALIZE_WHITESPACE
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
        ... ) # doctest: +NORMALIZE_WHITESPACE
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
        >>> print(saccade) # doctest: +NORMALIZE_WHITESPACE
        Saccade(name=saccade, onset=8, offset=10)
        """

        super().__init__(name=self._name, onset=onset, offset=offset)
