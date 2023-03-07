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
"""Test module pymovements.events.event_properties"""
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pymovements.events.event_properties import duration
from pymovements.events.event_properties import PROPERTIES


@pytest.mark.parametrize(
    ('input_df', 'exception', 'msg_substrings'),
    [
        pytest.param(
            pl.DataFrame(schema={'onset': pl.Int64}),
            pl.exceptions.ColumnNotFoundError,
            ('offset',),
            id='missing_offset_column',
        ),
        pytest.param(
            pl.DataFrame(schema={'offset': pl.Int64}),
            pl.exceptions.ColumnNotFoundError,
            ('onset',),
            id='missing_onset_column',
        ),
    ],
)
def test_duration_exceptions(input_df, exception, msg_substrings):
    with pytest.raises(exception) as excinfo:
        input_df.select([duration()])

    msg, = excinfo.value.args
    for msg_substring in msg_substrings:
        assert msg_substring.lower() in msg.lower()


@pytest.mark.parametrize(
    ('input_df', 'expected_df'),
    [
        pytest.param(
            pl.DataFrame(schema={'onset': pl.Int64, 'offset': pl.Int64}),
            pl.DataFrame(schema={'duration': pl.Int64}),
            id='empty_dataframe_results_in_empty_dataframe_with_correct_schema',
        ),
        pytest.param(
            pl.DataFrame({'onset': 0, 'offset': 1}, schema={'onset': pl.Int64, 'offset': pl.Int64}),
            pl.DataFrame({'duration': 1}, schema={'duration': pl.Int64}),
            id='single_event_duration_1',
        ),
        pytest.param(
            pl.DataFrame(
                {'onset': [0, 10], 'offset': [9, 23]},
                schema={'onset': pl.Int64, 'offset': pl.Int64},
            ),
            pl.DataFrame(
                {'duration': [9, 13]},
                schema={'duration': pl.Int64},
            ),
            id='two_events_different_durations',
        ),
    ],
)
def test_duration_has_expected_result(input_df, expected_df):
    result_df = input_df.select([duration()])

    assert_frame_equal(result_df, expected_df)


@pytest.mark.parametrize(
    'property_function',
    [
        pytest.param(duration),
    ],
)
def test_property_registered(property_function):
    assert property_function.__name__ in PROPERTIES
    assert PROPERTIES[property_function.__name__] == property_function
