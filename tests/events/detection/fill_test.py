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
"""This module tests functionality of the IVT algorithm."""
from __future__ import annotations

import numpy as np
import pytest
from polars.testing import assert_frame_equal

from pymovements.events.detection.fill import fill
from pymovements.events.events import EventDataFrame
from pymovements.gaze.transforms import pos2vel


@pytest.mark.parametrize(
    ('kwargs', 'expected_error'),
    [
        pytest.param(
            {
                'events': EventDataFrame(),
                'positions': None,
                'velocities': np.ones((100, 2)),
                'minimum_duration': 1,
            },
            ValueError,
            id='positions_none_raises_value_error',
        ),
        pytest.param(
            {
                'events': EventDataFrame(),
                'positions': np.ones((100, 2)),
                'velocities': None,
                'minimum_duration': 1,
            },
            ValueError,
            id='velocities_none_raises_value_error',
        ),
        pytest.param(
            {
                'events': EventDataFrame(),
                'positions': 1,
                'velocities': np.ones((100, 2)),
                'minimum_duration': 1,
            },
            ValueError,
            id='positions_not_array_like_raises_value_error',
        ),
        pytest.param(
            {
                'events': EventDataFrame(),
                'positions': np.ones((100, 2)),
                'velocities': 1,
                'minimum_duration': 1,
            },
            ValueError,
            id='velocities_not_array_like_raises_value_error',
        ),
        pytest.param(
            {
                'events': EventDataFrame(),
                'positions': np.ones(100),
                'velocities': np.ones((100, 2)),
                'minimum_duration': 1,
            },
            ValueError,
            id='positions_not_2d_array_raises_value_error',
        ),
        pytest.param(
            {
                'events': EventDataFrame(),
                'positions': np.ones((100, 2)),
                'velocities': np.ones(100),
                'minimum_duration': 1,
            },
            ValueError,
            id='velocities_not_2d_array_raises_value_error',
        ),
        pytest.param(
            {
                'events': EventDataFrame(),
                'positions': np.ones((100, 3)),
                'velocities': np.ones((100, 2)),
                'minimum_duration': 1,
            },
            ValueError,
            id='positions_not_2_elements_in_second_dimension_raises_value_error',
        ),
        pytest.param(
            {
                'events': EventDataFrame(),
                'positions': np.ones((100, 2)),
                'velocities': np.ones((100, 3)),
                'minimum_duration': 1,
            },
            ValueError,
            id='velocities_not_2_elements_in_second_dimension_raises_value_error',
        ),
        pytest.param(
            {
                'events': EventDataFrame(),
                'positions': np.ones((100, 2)),
                'velocities': np.ones((101, 2)),
                'minimum_duration': 1,
            },
            ValueError,
            id='positions_and_velocities_different_lengths_raises_value_error',
        ),
    ],
)
def test_fill_raise_error(kwargs, expected_error):
    """Test if ivt raises expected error."""
    with pytest.raises(expected_error):
        fill(**kwargs)


@pytest.mark.parametrize(
    ('kwargs', 'expected'),
    [
        pytest.param(
            {
                'events': EventDataFrame(name='fixation', onsets=[0], offsets=[100]),
                'positions': np.stack([np.arange(0, 100), np.arange(0, 100)], axis=1),
            },
            EventDataFrame(),
            id='fixation_from_start_to_end_no_fill',
        ),
        pytest.param(
            {
                'events': EventDataFrame(name='fixation', onsets=[10], offsets=[100]),
                'positions': np.stack([np.arange(0, 100), np.arange(0, 100)], axis=1),
            },
            EventDataFrame(
                name='unclassified',
                onsets=[0],
                offsets=[9],
            ),
            id='fixation_10_ms_after_start_to_end_single_fill',
        ),
        pytest.param(
            {
                'events': EventDataFrame(name='fixation', onsets=[0], offsets=[90]),
                'positions': np.stack([np.arange(0, 100), np.arange(0, 100)], axis=1),
            },
            EventDataFrame(
                name='unclassified',
                onsets=[90],
                offsets=[99],
            ),
            id='fixation_from_start_to_10_ms_before_end_single_fill',
        ),
        pytest.param(
            {
                'events': EventDataFrame(name='fixation', onsets=[0, 50], offsets=[40, 100]),
                'positions': np.stack([np.arange(0, 100), np.arange(0, 100)], axis=1),
            },
            EventDataFrame(
                name='unclassified',
                onsets=[40],
                offsets=[49],
            ),
            id='fixation_10_ms_break_at_40ms_single_fill',
        ),
        pytest.param(
            {
                'events': EventDataFrame(
                    name=['fixation', 'saccade'], onsets=[0, 50], offsets=[40, 100],
                ),
                'positions': np.stack([np.arange(0, 100), np.arange(0, 100)], axis=1),
            },
            EventDataFrame(
                name='unclassified',
                onsets=[40],
                offsets=[49],
            ),
            id='fixation_10_ms_break_then_saccade_until_end_single_fill',
        ),
    ],
)
def test_fill_fills_events(kwargs, expected):
    velocities = pos2vel(kwargs['positions'], sampling_rate=10, method='preceding')
    events = fill(velocities=velocities, **kwargs)

    assert_frame_equal(events.frame, expected.frame)
