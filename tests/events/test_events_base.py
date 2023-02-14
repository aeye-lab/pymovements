# Copyright (c) 2022 The pymovements Project Authors
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
This module tests functionality of base event classes.
"""
import pytest

from pymovements.events import Event
from pymovements.events import Fixation
from pymovements.events import Saccade


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
    ],
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
