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
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pymovements.datasets.dataset import Dataset
from pymovements.events.events import Event


@pytest.mark.parametrize(
    'events_init, events_expected',
    [
        pytest.param(
            [],
            [],
            id='empty_list_stays_empty_list',
        ),
        pytest.param(
            [pl.DataFrame(schema=Event.schema)],
            [pl.DataFrame(schema=Event.schema)],
            id='empty_df_stays_empty_df',
        ),
        pytest.param(
            [pl.DataFrame({'type': ['event'], 'onset': [0], 'offset': [99]}, schema=Event.schema)],
            [pl.DataFrame(schema=Event.schema)],
            id='single_instance_filled_df_gets_cleared_to_empty_df',
        ),
        pytest.param(
            [
                pl.DataFrame(
                    {'type': ['event'], 'onset': [0], 'offset': [99]},
                    schema=Event.schema,
                ),
                pl.DataFrame(
                    {'type': ['event'], 'onset': [0], 'offset': [99]},
                    schema=Event.schema,
                ),
            ],
            [pl.DataFrame(schema=Event.schema), pl.DataFrame(schema=Event.schema)],
            id='two_instance_filled_df_gets_cleared_to_two_empty_dfs',
        ),
    ],
)
def test_clear_events(events_init, events_expected):
    """Test if idt detects fixations."""
    dataset = Dataset(root='data')
    dataset.events = events_init
    dataset.clear_events()

    if isinstance(events_init, list) and not events_init:
        assert dataset.events == events_expected

    else:
        for events_df_result, events_df_expected in zip(dataset.events, events_expected):
            assert_frame_equal(events_df_result, events_df_expected)
