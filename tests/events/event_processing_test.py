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
"""Test event processing classes."""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pymovements.events.event_processing import EventGazeProcessor
from pymovements.events.event_processing import EventProcessor
from pymovements.events.events import EventDataFrame
from pymovements.exceptions import InvalidProperty
from pymovements.gaze.gaze_dataframe import GazeDataFrame


@pytest.mark.parametrize(
    ('args', 'kwargs', 'expected_property_definitions'),
    [
        pytest.param(['duration'], {}, ['duration'], id='arg_str_duration'),
        pytest.param([['duration']], {}, ['duration'], id='arg_list_duration'),
        pytest.param(
            [], {'event_properties': 'duration'}, ['duration'],
            id='kwarg_properties_duration',
        ),
    ],
)
def test_event_processor_init(args, kwargs, expected_property_definitions):
    processor = EventProcessor(*args, **kwargs)

    assert processor.event_properties == expected_property_definitions


@pytest.mark.parametrize(
    ('args', 'kwargs', 'exception', 'msg_substrings'),
    [
        pytest.param(
            ['foo'], {},
            InvalidProperty, ('foo', 'invalid', 'duration'),
            id='unknown_event_property',
        ),
    ],
)
def test_event_processor_init_exceptions(args, kwargs, exception, msg_substrings):
    with pytest.raises(exception) as excinfo:
        EventProcessor(*args, **kwargs)

    msg, = excinfo.value.args
    for msg_substring in msg_substrings:
        assert msg_substring.lower() in msg.lower()


@pytest.mark.parametrize(
    ('events_kwargs', 'event_properties', 'expected_dataframe'),
    [
        pytest.param(
            {'onsets': [], 'offsets': []},
            'duration',
            pl.DataFrame(schema={'duration': pl.Int64}),
            id='duration_no_event',
        ),
        pytest.param(
            {'onsets': [0], 'offsets': [1]},
            'duration',
            pl.DataFrame(data=[1], schema={'duration': pl.Int64}),

            id='duration_single_event',
        ),
        pytest.param(
            {'onsets': [0, 100], 'offsets': [1, 111]},
            'duration',
            pl.DataFrame(data=[1, 11], schema={'duration': pl.Int64}),
            id='duration_two_events',
        ),
    ],
)
def test_event_processor_process_correct_result(
        events_kwargs, event_properties, expected_dataframe,
):
    events = EventDataFrame(**events_kwargs)
    processor = EventProcessor(event_properties)

    property_result = processor.process(events)
    assert_frame_equal(property_result, expected_dataframe)


@pytest.mark.parametrize(
    ('args', 'kwargs', 'expected_property_definitions'),
    [
        pytest.param(['peak_velocity'], {}, ['peak_velocity'], id='arg_str_peak_velocity'),
        pytest.param([['peak_velocity']], {}, ['peak_velocity'], id='arg_list_peak_velocity'),
        pytest.param(
            [], {'event_properties': 'peak_velocity'}, ['peak_velocity'],
            id='kwarg_properties_peak_velocity',
        ),
    ],
)
def test_event_gaze_processor_init(args, kwargs, expected_property_definitions):
    processor = EventGazeProcessor(*args, **kwargs)

    assert processor.event_properties == expected_property_definitions


@pytest.mark.parametrize(
    ('args', 'kwargs', 'exception', 'msg_substrings'),
    [
        pytest.param(
            ['foo'], {},
            InvalidProperty, ('foo', 'invalid', 'peak_velocity'),
            id='unknown_event_property',
        ),
    ],
)
def test_event_gaze_processor_init_exceptions(args, kwargs, exception, msg_substrings):
    with pytest.raises(exception) as excinfo:
        EventGazeProcessor(*args, **kwargs)

    msg, = excinfo.value.args
    for msg_substring in msg_substrings:
        assert msg_substring.lower() in msg.lower()


@pytest.mark.parametrize(
    ('event_df', 'gaze_df', 'init_kwargs', 'process_kwargs', 'expected_dataframe'),
    [
        pytest.param(
            pl.from_dict(
                {'subject_id': [1], 'onset': [0], 'offset': [10]},
                schema={'subject_id': pl.Int64, 'onset': pl.Int64, 'offset': pl.Int64},
            ),
            pl.from_dict(
                {
                    'subject_id': np.ones(10),
                    'time': np.arange(10),
                    'x_vel': np.ones(10),
                    'y_vel': np.zeros(10),
                },
                schema={
                    'subject_id': pl.Int64,
                    'time': pl.Int64,
                    'x_vel': pl.Float64,
                    'y_vel': pl.Float64,
                },
            ),
            {'event_properties': 'peak_velocity'},
            {'identifiers': 'subject_id'},
            pl.from_dict(
                {
                    'subject_id': [1],
                    'name': [None],
                    'onset': [0],
                    'offset': [10],
                    'peak_velocity': [1],
                },
                schema={
                    'subject_id': pl.Int64,
                    'name': pl.Utf8,
                    'onset': pl.Int64,
                    'offset': pl.Int64,
                    'peak_velocity': pl.Float64,
                },
            ),
            id='peak_velocity_single_event_complete_window',
        ),
        pytest.param(
            pl.from_dict(
                {'subject_id': [1], 'onset': [0], 'offset': [10]},
                schema={'subject_id': pl.Int64, 'onset': pl.Int64, 'offset': pl.Int64},
            ),
            pl.from_dict(
                {
                    'subject_id': np.ones(10),
                    'time': np.arange(10),
                    'x_vel': np.ones(10),
                    'y_vel': np.zeros(10),
                },
                schema={
                    'subject_id': pl.Int64,
                    'time': pl.Int64,
                    'x_vel': pl.Float64,
                    'y_vel': pl.Float64,
                },
            ),
            {'event_properties': 'peak_velocity'},
            {'identifiers': 'subject_id'},
            pl.from_dict(
                {
                    'subject_id': [1],
                    'name': [None],
                    'onset': [0],
                    'offset': [10],
                    'peak_velocity': [1],
                },
                schema={
                    'subject_id': pl.Int64,
                    'name': pl.Utf8,
                    'onset': pl.Int64,
                    'offset': pl.Int64,
                    'peak_velocity': pl.Float64,
                },
            ),
            id='peak_velocity_custom_columns_single_event_complete_window',
        ),
        pytest.param(
            pl.from_dict(
                {'subject_id': [1], 'onset': [0], 'offset': [5]},
                schema={'subject_id': pl.Int64, 'onset': pl.Int64, 'offset': pl.Int64},
            ),
            pl.from_dict(
                {
                    'subject_id': np.ones(10),
                    'time': np.arange(10),
                    'x_vel': np.concatenate([np.ones(5), np.zeros(5)]),
                    'y_vel': np.zeros(10),
                },
                schema={
                    'subject_id': pl.Int64,
                    'time': pl.Int64,
                    'x_vel': pl.Float64,
                    'y_vel': pl.Float64,
                },
            ),
            {'event_properties': 'peak_velocity'},
            {'identifiers': 'subject_id'},
            pl.from_dict(
                {
                    'subject_id': [1],
                    'name': [None],
                    'onset': [0],
                    'offset': [5],
                    'peak_velocity': [1],
                },
                schema={
                    'subject_id': pl.Int64,
                    'name': pl.Utf8,
                    'onset': pl.Int64,
                    'offset': pl.Int64,
                    'peak_velocity': pl.Float64,
                },
            ),
            id='peak_velocity_single_event_half_window',
        ),
        pytest.param(
            pl.from_dict(
                {'subject_id': [1], 'onset': [0], 'offset': [10]},
                schema={'subject_id': pl.Int64, 'onset': pl.Int64, 'offset': pl.Int64},
            ),
            pl.from_dict(
                {
                    'subject_id': np.ones(10),
                    'time': np.arange(10),
                    'x_pos': np.concatenate([np.ones(5), np.zeros(5)]),
                    'y_pos': np.concatenate([np.zeros(5), np.ones(5)]),
                },
                schema={
                    'subject_id': pl.Int64,
                    'time': pl.Int64,
                    'x_pos': pl.Float64,
                    'y_pos': pl.Float64,
                },
            ),
            {'event_properties': 'dispersion'},
            {'identifiers': 'subject_id'},
            pl.from_dict(
                {
                    'subject_id': [1],
                    'name': [None],
                    'onset': [0],
                    'offset': [10],
                    'dispersion': [2],
                },
                schema={
                    'subject_id': pl.Int64,
                    'name': pl.Utf8,
                    'onset': pl.Int64,
                    'offset': pl.Int64,
                    'dispersion': pl.Float64,
                },
            ),
            id='dispersion_single_event_complete_window',
        ),
    ],
)
def test_event_gaze_processor_process_correct_result(
        event_df, gaze_df, init_kwargs, process_kwargs, expected_dataframe,
):
    processor = EventGazeProcessor(**init_kwargs)
    events = EventDataFrame(event_df)
    gaze = GazeDataFrame(gaze_df)
    property_result = processor.process(events, gaze, **process_kwargs)
    assert_frame_equal(property_result, expected_dataframe)


@pytest.mark.parametrize(
    ('event_df', 'gaze_df', 'init_kwargs', 'process_kwargs', 'exception', 'msg_substrings'),
    [
        pytest.param(
            pl.from_dict(
                {'subject_id': [1], 'onset': [0], 'offset': [10]},
                schema={'subject_id': pl.Int64, 'onset': pl.Int64, 'offset': pl.Int64},
            ),
            pl.from_dict(
                {
                    'subject_id': np.ones(10),
                    'time': np.arange(10),
                    'x_vel': np.ones(10),
                    'y_vel': np.zeros(10),
                },
                schema={
                    'subject_id': pl.Int64,
                    'time': pl.Int64,
                    'x_vel': pl.Float64,
                    'y_vel': pl.Float64,
                },
            ),
            {'event_properties': 'peak_velocity'},
            {'identifiers': []},
            ValueError,
            ('identifiers', 'list', 'must', 'not', 'empty'),
            id='empty_list',
        ),
    ],
)
def test_event_processor_process_exceptions(
        event_df, gaze_df, init_kwargs, process_kwargs, exception, msg_substrings,
):
    processor = EventGazeProcessor(**init_kwargs)
    events = EventDataFrame(event_df)
    gaze = GazeDataFrame(gaze_df)

    with pytest.raises(exception) as excinfo:
        processor.process(events, gaze, **process_kwargs)

    msg, = excinfo.value.args
    for msg_substring in msg_substrings:
        assert msg_substring.lower() in msg.lower()
