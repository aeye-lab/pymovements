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
"""Tests pymovements.events.events.EventDataFrame."""
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pymovements import exceptions
from pymovements.events.events import EventDataFrame


@pytest.fixture(name='minimal_schema')
def fixture_dataset():
    schema = {'name': pl.Utf8, 'onset': pl.Int64, 'offset': pl.Int64}
    yield schema


@pytest.mark.parametrize(
    ('kwargs', 'exception', 'msg_substrings'),
    [
        pytest.param(
            {'onsets': None, 'offsets': []},
            ValueError, ('onsets', 'offsets', 'both None', 'or', 'both not None'),
            id='onsets_none_offsets_list',
        ),
        pytest.param(
            {'onsets': [], 'offsets': None},
            ValueError, ('onsets', 'offsets', 'both None', 'or', 'both not None'),
            id='onsets_list_offsets_none',
        ),
        pytest.param(
            {'onsets': [], 'offsets': [0]},
            ValueError, ('onsets', 'offsets', 'length', 'equal'),
            id='onsets_empty_list_offsets_single_int',
        ),
        pytest.param(
            {'onsets': [1], 'offsets': []},
            ValueError, ('onsets', 'offsets', 'length', 'equal'),
            id='onsets_single_int_offsets_empty_list',
        ),
        pytest.param(
            {'data': pl.DataFrame(), 'name': None, 'onsets': 1, 'offsets': None},
            ValueError, ('data', 'onsets', 'mutually', 'exclusive'),
            id='data_with_onsets_raises_mutually_exclusive',
        ),
        pytest.param(
            {'data': pl.DataFrame(), 'offsets': 1},
            ValueError, ('data', 'offsets', 'mutually', 'exclusive'),
            id='data_with_offsets_raises_mutually_exclusive',
        ),
        pytest.param(
            {'data': pl.DataFrame(), 'name': 1},
            ValueError, ('data', 'name', 'mutually', 'exclusive'),
            id='data_with_name_raises_mutually_exclusive',
        ),
    ],
)
def test_event_dataframe_init_exceptions(kwargs, exception, msg_substrings):
    with pytest.raises(exception) as excinfo:
        EventDataFrame(**kwargs)

    msg, = excinfo.value.args
    for msg_substring in msg_substrings:
        assert msg_substring in msg


@pytest.mark.parametrize(
    ('args', 'kwargs'),
    [
        pytest.param([], {}, id='no_args_no_kwargs'),
        pytest.param([], {'onsets': [], 'offsets': []}, id='dict_with_empty_lists_kwarg'),
        pytest.param([], {'onsets': [0], 'offsets': [1]}, id='dict_with_single_event_kwarg'),
        pytest.param([], {'onsets': [0, 2], 'offsets': [1, 3]}, id='dict_with_two_events_kwarg'),
    ],
)
def test_event_dataframe_init_minimal_schema(args, kwargs, minimal_schema):
    event_df = EventDataFrame(*args, **kwargs)
    assert event_df.schema == minimal_schema


@pytest.mark.parametrize(
    ('args', 'kwargs', 'expected_length'),
    [
        pytest.param([], {}, 0, id='no_args_no_kwargs'),
        pytest.param([], {'onsets': [], 'offsets': []}, 0, id='dict_with_empty_lists_kwarg'),
        pytest.param([], {'onsets': [0], 'offsets': [1]}, 1, id='dict_with_single_event_kwarg'),
        pytest.param([], {'onsets': [0, 2], 'offsets': [1, 3]}, 2, id='dict_with_two_events_kwarg'),
    ],
)
def test_event_dataframe_init_has_expected_length(args, kwargs, expected_length):
    event_df = EventDataFrame(*args, **kwargs)
    assert len(event_df) == expected_length


@pytest.mark.parametrize(
    ('args', 'kwargs', 'expected_name'),
    [
        pytest.param(
            [pl.DataFrame()], {}, 'foo',
            id='dataframe_arg_dict_with_single_event_kwarg',
        ),
        pytest.param(
            [], {'name': 'bar', 'onsets': [0], 'offsets': [1]}, 'bar',
            id='dict_with_single_event_with_name_kwarg',
        ),
        pytest.param(
            [], {'name': 'bar', 'onsets': [0, 1], 'offsets': [1, 2]}, 'bar',
            id='dict_with_two_events_with_name_kwarg',
        ),
    ],
)
def test_event_dataframe_init_has_correct_name(args, kwargs, expected_name):
    event_df = EventDataFrame(*args, **kwargs)
    assert (event_df['name'].to_numpy() == expected_name).all()


@pytest.mark.parametrize(
    ('args', 'kwargs', 'expected_names'),
    [
        pytest.param(
            [], {'name': ['foo', 'bar'], 'onsets': [0, 1], 'offsets': [1, 2]}, ['foo', 'bar'],
            id='dict_with_two_events_with_name_kwarg',
        ),
    ],
)
def test_event_dataframe_init_has_correct_names(args, kwargs, expected_names):
    event_df = EventDataFrame(*args, **kwargs)
    assert (event_df['name'] == expected_names).all()


@pytest.mark.parametrize(
    ('args', 'kwargs', 'expected_df_data'),
    [
        pytest.param(
            [], {'onsets': [0], 'offsets': [1]},
            {'name': [''], 'onset': [0], 'offset': [1]},
            id='no_arg_dict_with_single_event_kwarg',
        ),
        pytest.param(
            [pl.DataFrame()], {},
            {},
            id='dataframe_arg_no_kwargs',
        ),
        pytest.param(
            [], {'name': 'bar', 'onsets': [0], 'offsets': [1]},
            {'name': ['bar'], 'onset': [0], 'offset': [1]},
            id='dict_with_single_named_event',
        ),
        pytest.param(
            [], {'name': 'bar', 'onsets': [0, 2], 'offsets': [1, 3]},
            {'name': ['bar', 'bar'], 'onset': [0, 2], 'offset': [1, 3]},
            id='dict_with_two_events_same_name',
        ),
        pytest.param(
            [], {'name': ['foo', 'bar'], 'onsets': [0, 2], 'offsets': [1, 3]},
            {'name': ['foo', 'bar'], 'onset': [0, 2], 'offset': [1, 3]},
            id='dict_with_two_differently_named_events',
        ),
    ],
)
def test_event_dataframe_init_expected(args, kwargs, expected_df_data, minimal_schema):
    event_df = EventDataFrame(*args, **kwargs)

    expected_df = pl.DataFrame(data=expected_df_data, schema=minimal_schema)
    assert_frame_equal(event_df.frame, expected_df)


@pytest.mark.parametrize(
    ('init_kwargs', 'property_kwargs', 'exception', 'msg_substrings'),
    [
        pytest.param(
            {},
            {'property_name': 'foo'},
            exceptions.InvalidProperty,
            ('foo', 'invalid', 'valid', 'duration'),
            id='invalid_property',
        ),
    ],
)
def test_event_dataframe_add_property_raises_exceptions(
        init_kwargs, property_kwargs, exception, msg_substrings,
):
    event_df = EventDataFrame(**init_kwargs)

    with pytest.raises(exception) as excinfo:
        event_df.add_property(**property_kwargs)

    msg, = excinfo.value.args
    for msg_substring in msg_substrings:
        assert msg_substring.lower() in msg.lower()


@pytest.mark.parametrize(
    ('init_kwargs', 'property_kwargs', 'expected_df'),
    [
        pytest.param(
            {'onsets': [0], 'offsets': [1]},
            {'property_name': 'duration'},
            EventDataFrame(pl.DataFrame({'name': '', 'onset': 0, 'offset': 1, 'duration': 1})),
            id='single_event_duration_property',
        ),
        pytest.param(
            {'onsets': [0, 100], 'offsets': [1, 111]},
            {'property_name': 'duration'},
            EventDataFrame(
                pl.DataFrame(
                    {'name': ['', ''], 'onset': [0, 100], 'offset': [1, 111], 'duration': [1, 11]},
                ),
            ),
            id='two_events_duration_property',
        ),
    ],
)
def test_event_dataframe_add_property_has_expected_result(
        init_kwargs, property_kwargs, expected_df,
):
    event_df = EventDataFrame(**init_kwargs)
    event_df.add_property(**property_kwargs)

    assert_frame_equal(event_df.frame, expected_df.frame)
