"""
This module tests functionality of base event classes.
"""
from __future__ import annotations

import pytest

from pymovements.events import Event
from pymovements.events import Fixation
from pymovements.events import Saccade


# pylint: disable=protected-access
@pytest.mark.parametrize(
    'event_class, init_params, expected',
    [
        pytest.param(
            Event,
            {'name': 'custom_event', 'onset': 10, 'offset': 11},
            {'name': 'custom_event', 'duration': 1},
            id='custom_event_short',
        ),
        pytest.param(
            Event,
            {'name': 'custom_event', 'onset': 0, 'offset': 999},
            {'name': 'custom_event', 'duration': 999},
            id='custom_event_long',
        ),
        pytest.param(
            Fixation,
            {'onset': 10, 'offset': 100, 'position': (1, 1)},
            {'name': Fixation._name, 'duration': 90, 'position': (1, 1)},
            id='fixation',
        ),
        pytest.param(
            Saccade,
            {'onset': 10, 'offset': 100},
            {'name': Saccade._name, 'duration': 90},
            id='saccade',
        ),
    ]
)
def test_event_class(event_class, init_params, expected):
    """Test if instantiated event attributes fit expected values."""
    event = event_class(**init_params)

    assert event.name == expected['name'], (
        f'event name does not match expected value ({event.name} != {expected["name"]})'
    )
    assert event.duration == expected['duration'], (
        f'event duration does not match expected value ({event.duration} != {expected["duration"]})'
    )
    if 'position' in expected.keys():
        assert event.position == expected['position'], (
            'fixation event position does not match expected value'
            f'({event.position} != {expected["position"]})'
        )
